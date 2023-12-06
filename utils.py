def truncate_positions(positions, max_seq_len):
    if positions.size(0) > max_seq_len:
        positions = positions[:max_seq_len]
    return positions
