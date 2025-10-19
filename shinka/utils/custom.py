import random

def truncate_log_blocks(stderr_log: str, max_bytes: int = 200000, num_blocks: int = 10) -> str:
    """
    Truncate a large stderr log by removing random blocks of text until it fits `max_bytes`.
    Inserts "[TRUNCATED]" where chunks were deleted and at the end.

    Args:
        stderr_log (str): Full stderr log.
        max_bytes (int): Maximum allowed size in bytes (default: 200 KB).
        num_blocks (int): Number of random blocks to remove (default: 10).

    Returns:
        str: Truncated stderr log with "[TRUNCATED]" markers.
    """
    log_bytes = stderr_log.encode("utf-8")
    n = len(log_bytes)
    if n <= max_bytes:
        return stderr_log

    target_size = int(max_bytes * 0.95)  # leave room for markers
    remove_bytes = n - target_size
    block_size = remove_bytes // num_blocks
    if block_size < 1:
        block_size = 1

    # Choose random start positions for removal blocks
    #starts = sorted(random.sample(range(0, max(1, n - block_size)), num_blocks))
    interval = n // num_blocks
    starts = [random.randint(i * interval, min((i + 1) * interval - block_size, n - block_size))
              for i in range(num_blocks)]
    starts = sorted(starts)
    segments = []
    cursor = 0

    for start in starts:
        end = min(start + block_size, n)
        if start > cursor:
            segments.append(log_bytes[cursor:start])
            segments.append(b"\n[TRUNCATED]\n")
        cursor = end

    # Add remaining tail
    if cursor < n:
        segments.append(log_bytes[cursor:])

    segments.append(b"\n[TRUNCATED]\n")

    result = b"".join(segments)

    return result.decode("utf-8", errors="replace")
