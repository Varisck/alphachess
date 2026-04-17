"""Move encoding and board encoding

Action space: 8×8×73 = 4672 actions:
  source_square = rank * 8 + file
  index = source_square * 73 + plane
  plane ∈ [0, 72]: queen-style (0–55), knight (56–63), underpromotion (64–72)
"""

from __future__ import annotations

import chess
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_ACTIONS = 4672
_PLANES_PER_SQ = 73

# Queen-style direction index → (Δfile, Δrank)
_DIRECTION_OFFSETS: list[tuple[int, int]] = [
    (0,  1),   # 0: N
    (1,  1),   # 1: NE
    (1,  0),   # 2: E
    (1, -1),   # 3: SE
    (0, -1),   # 4: S
    (-1, -1),  # 5: SW
    (-1,  0),  # 6: W
    (-1,  1),  # 7: NW
]

# (Δfile_sign, Δrank_sign) → direction index
_DIRECTION_MAP: dict[tuple[int, int], int] = {
    v: k for k, v in enumerate(_DIRECTION_OFFSETS)
}

# Knight plane offsets (Δfile, Δrank), index = plane - 56
_KNIGHT_OFFSETS: list[tuple[int, int]] = [
    ( 1,  2),  # 56: NNE
    ( 2,  1),  # 57: ENE
    ( 2, -1),  # 58: ESE
    ( 1, -2),  # 59: SSE
    (-1, -2),  # 60: SSW
    (-2, -1),  # 61: WSW
    (-2,  1),  # 62: WNW
    (-1,  2),  # 63: NNW
]

_KNIGHT_PLANE: dict[tuple[int, int], int] = {
    offset: 56 + i for i, offset in enumerate(_KNIGHT_OFFSETS)
}

# Underpromotion piece order: knight=0, bishop=1, rook=2
_UNDERPROMO_PIECES: list[int] = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
_UNDERPROMO_IDX: dict[int, int] = {p: i for i, p in enumerate(_UNDERPROMO_PIECES)}

# Piece plane order, same for both colors (planes 0–5 own, 6–11 opponent)
_PIECE_TYPES: list[int] = [
    chess.PAWN, chess.KNIGHT, chess.BISHOP,
    chess.ROOK, chess.QUEEN, chess.KING,
]


# ---------------------------------------------------------------------------
# Square helpers
# ---------------------------------------------------------------------------

def _flip_sq(sq: int) -> int:
    # vertical flip: rank → 7-rank, file unchanged
    return (7 - sq // 8) * 8 + (sq % 8)


# ---------------------------------------------------------------------------
# Board → [18, 8, 8] tensor
# ---------------------------------------------------------------------------

def encode(board: chess.Board) -> np.ndarray:
    """Encode board state as [18, 8, 8] float32 from the current player's view.

    Planes 0–5:  current player's pieces (P N B R Q K)
    Planes 6–11: opponent's pieces (P N B R Q K)
    Planes 12–15: castling rights (our K, our Q, their K, their Q)
    Plane 16:    en passant target square
    Plane 17:    fifty-move counter / 100
    """
    planes = np.zeros((18, 8, 8), dtype=np.float32)
    flip = board.turn == chess.BLACK

    our_color = chess.BLACK if flip else chess.WHITE
    their_color = chess.WHITE if flip else chess.BLACK

    # Piece planes 0-5 (ours) and 6-11 (theirs)
    for i, pt in enumerate(_PIECE_TYPES):
        for plane_idx, color in [(i, our_color), (i + 6, their_color)]:
            squares = np.fromiter(board.pieces(pt, color), dtype=np.int8)
            if squares.size == 0:
                continue
            if flip:
                squares = (7 - squares // 8) * 8 + squares % 8
            ranks, files = np.divmod(squares, 8)
            planes[plane_idx, ranks, files] = 1.0

    # Castling rights: 12 = our kingside, 13 = our queenside,
    #                  14 = their kingside, 15 = their queenside
    if flip:
        planes[12] = float(board.has_kingside_castling_rights(chess.BLACK))
        planes[13] = float(board.has_queenside_castling_rights(chess.BLACK))
        planes[14] = float(board.has_kingside_castling_rights(chess.WHITE))
        planes[15] = float(board.has_queenside_castling_rights(chess.WHITE))
    else:
        planes[12] = float(board.has_kingside_castling_rights(chess.WHITE))
        planes[13] = float(board.has_queenside_castling_rights(chess.WHITE))
        planes[14] = float(board.has_kingside_castling_rights(chess.BLACK))
        planes[15] = float(board.has_queenside_castling_rights(chess.BLACK))

    # En passant target square (already in real coords; flip rank if Black to move)
    if board.ep_square is not None:
        ep_sq = _flip_sq(board.ep_square) if flip else board.ep_square
        r, f = divmod(ep_sq, 8)
        planes[16, r, f] = 1.0

    # Fifty-move counter, normalized
    planes[17] = board.halfmove_clock / 100.0

    return planes


# ---------------------------------------------------------------------------
# Move ↔ action index
# ---------------------------------------------------------------------------

def move_to_index(move: chess.Move, board: chess.Board) -> int:
    """Convert a legal chess.Move to its 0–4671 action index.

    Applies canonical orientation: if Black to move, both squares are flipped
    vertically before the plane is computed.
    """
    flip = board.turn == chess.BLACK

    from_sq = _flip_sq(move.from_square) if flip else move.from_square
    to_sq   = _flip_sq(move.to_square)   if flip else move.to_square

    from_rank, from_file = divmod(from_sq, 8)
    to_rank,   to_file   = divmod(to_sq, 8)
    d_rank = to_rank - from_rank
    d_file = to_file - from_file

    # Underpromotion (queen promotion falls through to queen-style)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        piece_idx = _UNDERPROMO_IDX[move.promotion]
        if d_file == 0:
            dir_idx = 0   # forward
        elif d_file < 0:
            dir_idx = 1   # capture-left
        else:
            dir_idx = 2   # capture-right
        plane = 64 + piece_idx * 3 + dir_idx
        return from_sq * _PLANES_PER_SQ + plane

    # Knight move
    knight_plane = _KNIGHT_PLANE.get((d_file, d_rank))
    if knight_plane is not None:
        return from_sq * _PLANES_PER_SQ + knight_plane

    # Queen-style: rook, bishop, queen, king, pawn push/capture, queen promo
    sign_file = 0 if d_file == 0 else (1 if d_file > 0 else -1)
    sign_rank = 0 if d_rank == 0 else (1 if d_rank > 0 else -1)
    direction = _DIRECTION_MAP[(sign_file, sign_rank)]
    distance  = max(abs(d_rank), abs(d_file))
    plane = direction * 7 + (distance - 1)
    return from_sq * _PLANES_PER_SQ + plane


def index_to_move(index: int, board: chess.Board) -> chess.Move:
    """Convert action index to chess.Move.

    Applies canonical orientation in reverse: squares are unflipped for
    Black to move so the returned Move uses real-board coordinates.
    Raises ValueError if index is out of range.
    """
    if not (0 <= index < NUM_ACTIONS):
        raise ValueError(f"Action index out of range: {index}")

    flip = board.turn == chess.BLACK
    from_sq_can = index // _PLANES_PER_SQ
    plane       = index % _PLANES_PER_SQ

    from_sq = _flip_sq(from_sq_can) if flip else from_sq_can
    from_rank_can, from_file_can = divmod(from_sq_can, 8)

    if plane < 56:
        # Queen-style
        direction = plane // 7
        distance  = plane % 7 + 1
        d_file_can, d_rank_can = _DIRECTION_OFFSETS[direction]
        to_rank_can = from_rank_can + d_rank_can * distance
        to_file_can = from_file_can + d_file_can * distance
        to_sq_can   = to_rank_can * 8 + to_file_can
        to_sq = _flip_sq(to_sq_can) if flip else to_sq_can

        # implicit queen promotion: pawn reaching canonical back rank (rank 7)
        promotion = None
        piece = board.piece_at(from_sq)
        if piece is not None and piece.piece_type == chess.PAWN and to_rank_can == 7:
            promotion = chess.QUEEN

        return chess.Move(from_sq, to_sq, promotion=promotion)

    elif plane < 64:
        # Knight
        d_file_can, d_rank_can = _KNIGHT_OFFSETS[plane - 56]
        to_rank_can = from_rank_can + d_rank_can
        to_file_can = from_file_can + d_file_can
        to_sq_can   = to_rank_can * 8 + to_file_can
        to_sq = _flip_sq(to_sq_can) if flip else to_sq_can
        return chess.Move(from_sq, to_sq)

    else:
        # Underpromotion: planes 64–72
        up = plane - 64
        promotion = _UNDERPROMO_PIECES[up // 3]
        dir_idx   = up % 3
        d_file_can = [0, -1, 1][dir_idx]
        to_rank_can = from_rank_can + 1         # always one rank forward (canonical)
        to_file_can = from_file_can + d_file_can
        to_sq_can   = to_rank_can * 8 + to_file_can
        to_sq = _flip_sq(to_sq_can) if flip else to_sq_can
        return chess.Move(from_sq, to_sq, promotion=promotion)


def legal_action_mask(board: chess.Board) -> np.ndarray:
    """Return bool [4672] with True for each legal move.

    Caller should cache this per MCTS node — recomputing every visit is expensive.
    """
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = True
    return mask
