from __future__ import annotations

import chess
import numpy as np

from collections import deque

from alphachess.config import MCTSConfig
from alphachess.game.encoding import legal_action_mask, index_to_move

class Tree:
    """Array structure of the MCTS' tree for a single game.

    All per-node data lives in parallel numpy arrays indexed by node_id.
    Node 0 is always the root. New nodes are allocated sequentially.
    """

    def __init__(self, max_nodes: int, config: MCTSConfig, action_space: int = 4672) -> None:
        self._config = config
        self._action_space = action_space

        # Visit counts and accumulated values — shape [max_nodes, action_space]
        self.N = np.zeros((max_nodes, action_space), dtype=np.int32)
        self.W = np.zeros((max_nodes, action_space), dtype=np.float32)

        # Prior probabilities from the network — shape [max_nodes, action_space]
        self.P = np.zeros((max_nodes, action_space), dtype=np.float32)

        # Child node indices; -1 means child not yet allocated
        # shape [max_nodes, action_space]
        self.children = np.full((max_nodes, action_space), -1, dtype=np.int32)

        # Whether expand() has been called for this node
        self.is_expanded = np.zeros(max_nodes, dtype=bool)

        # Board state at each node (set by expand)
        self.boards: list[chess.Board | None] = [None] * max_nodes

        # Legal move mask — shape [max_nodes, action_space]
        self.legal_masks = np.zeros((max_nodes, action_space), dtype=bool)

        # Navigation: parent node and the action taken to reach this node.
        # Root (node 0) has parent = -1 and parent_action = -1.
        self.parent = np.full(max_nodes, -1, dtype=np.int32)
        self.parent_action = np.full(max_nodes, -1, dtype=np.int32)

        # Next free node slot; root is pre-allocated as node 0.
        self.n_nodes: int = 1

    # ------------------------------------------------------------------
    # Core operations (called by MCTS.run and the selfplay worker)
    # ------------------------------------------------------------------

    def descend_to_leaf(self, root_board: chess.Board) -> tuple[chess.Board, int]:
        """Walk from root to an unexpanded node using PUCT selection.

        Args:
            root_board: current board at the root 

        Returns
        -------
            pair: (leaf_board, leaf_node_id) - the board and node index of the first 
            unexpanded leaf.
        """
        node_id = 0
        board = self.boards[0] if self.boards[0] is not None else root_board
        while self.is_expanded[node_id]:
            action = int(np.argmax(self._puct_scores(node_id)))
            if self.children[node_id, action] == -1:
                self.children[node_id, action] = self._allocate_node(node_id, action)
            
            board = board.copy()
            board.push(index_to_move(action, board))
            node_id = self.children[node_id, action]
            
        return (board, node_id)



    def expand(self, node_id: int, board: chess.Board, priors: np.ndarray) -> None:
        """Store the network output at node_id and mark it as expanded.

        Args:
            node_id (int): node to expand.
            board (np.ndarray): board state at this node
                (returned by descend_to_leaf).
            priors (np.ndarray): [action_space] softmax network output
                over the full action space.
        """
        self.boards[node_id] = board
        self.legal_masks[node_id] = legal_action_mask(board)
        priors = priors * self.legal_masks[node_id]
        # check the np.sum is not 0 if else to do
        self.P[node_id] = priors / np.sum(priors)
        self.is_expanded[node_id] = True


    def backup(self, node_id: int, value: float) -> None:
        """Propagate value from node_id up to the root.

        Args:
            node_id: leaf node where evaluation started.
            value: network score in [-1, 1], from the perspective 
                of the player to move at node_id.
        """
        while node_id > 0:     # when hitting root node_id == -1
            parent = self.parent[node_id]
            action = self.parent_action[node_id]
            
            value = -value

            # increment visit count and value
            self.N[parent, action] += 1
            self.W[parent, action] += value

            node_id = parent


    def reroot(self, action: int) -> None:
        """Promote the child reached by ``action`` to become the new root.

        Preserves the subtree under that child (visits, priors, and all descendants) 
        and discards every other branch.

        Detail: the re-rooted tree uses node space, new simulations can add
        up to num_simulations nodes so max_nodes must account for that
        """
        
        new_root_id = self.children[0, action]
        max_nodes = self.N.shape[0]

        # if action was not explored rebuild new tree
        if new_root_id == -1:
            self.__init__(max_nodes, self._config)
            return
        
        # get the indicies of the reachable nodes from the root
        reachable = np.full(self.n_nodes, -1, np.int32)
        reachable[0] = new_root_id
        end = 1
        queue = deque([new_root_id])

        while queue:
            rn = int(queue.popleft())
            childrens = self.children[rn]
            childrens = childrens[childrens != -1].tolist()
            reachable[end:end + len(childrens)] = childrens
            end = end + len(childrens)
            queue.extend(childrens)

        indicies = reachable[:end]
        
        # inverse lookup 
        old_to_new = np.full(self.n_nodes + 1, -1, dtype=np.int32)
        old_to_new[indicies] = np.arange(end, dtype=np.int32)

        self.N[:end] = self.N[indicies]
        self.W[:end] = self.W[indicies]
        self.P[:end] = self.P[indicies]
        self.is_expanded[:end] = self.is_expanded[indicies]
        self.legal_masks[:end] = self.legal_masks[indicies]
        self.parent_action[:end] = self.parent_action[indicies]

        # children remap entries
        self.children[:end] = old_to_new[self.children[indicies]]

        # parentremap entries
        self.parent[:end] = old_to_new[self.parent[indicies]]
        self.parent[0] = -1
        self.parent_action[0] = -1

        # Clear the tail slots
        self.N[end:] = 0
        self.W[end:] = 0
        self.P[end:] = 0
        self.children[end:] = -1
        self.is_expanded[end:] = False
        self.legal_masks[end:] = False
        self.parent[end:] = -1
        self.parent_action[end:] = -1
        max_nodes = self.N.shape[0]
        self.boards = [self.boards[i] for i in indicies] + [None] * (max_nodes - end)
        self.n_nodes = end



    def root_visit_distribution(self) -> np.ndarray:
        """Return the visit-count policy at the root.

        Returns:
            Distribution: [action_space]: float32, distribution of the moves
        """
        dist = self.N[0].astype(np.float32)
        return dist / dist.sum()


    def root_visits(self) -> int:
        """Total visit count across all actions at the root."""
        return int(np.sum(self.N[0]))
    

    def root_value(self) -> float:
        """Return the value at the root"""
        if self.root_visits() <= 0:
            return 0.0
        return float(self.W[0].sum() / self.root_visits())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _puct_scores(self, node_id: int) -> np.ndarray:
        """Compute PUCT scores for every action at node_id.

        Returns:
            PUCT scores: [action_space]: float32
        """
        N = self.N[node_id]
        W = self.W[node_id]
        P = self.P[node_id]
        mask = self.legal_masks[node_id]

        with np.errstate(invalid="ignore", divide="ignore"):    # avoids warning
            Q = np.where(N > 0, W / N, 0.0).astype(np.float32)   

        total_N = int(N.sum())     # total visits to node_id
        U = self._config.c_puct * P * (np.sqrt(total_N) / (1 + N))

        scores = Q + U
        scores[~mask] = -np.inf
        return scores


    def _allocate_node(self, parent_id: int, action: int) -> int:
        """Claim the next free node slot, record parent linkage, return node_id.
        
        Returns:
            Node_id: int
        """
        node_id = self.n_nodes
        # check if this is really needed maybe caller checks before doing this?
        assert node_id < len(self.parent), (
            f"Tree overflow: tried to allocate node {node_id} but max_nodes={len(self.parent)}"
        )
        self.n_nodes += 1
        self.parent[node_id] = parent_id
        self.parent_action[node_id] = action
        return node_id