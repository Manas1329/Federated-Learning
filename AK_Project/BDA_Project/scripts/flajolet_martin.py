import hashlib

def get_trailing_zeros(n):
    if n == 0:
        return 0
    count = 0
    while (n & 1) == 0:
        count += 1
        n >>= 1
    return count

def flajolet_martin(data_stream, num_hashes=10):
    max_zeros = [0] * num_hashes

    for item in data_stream:
        for i in range(num_hashes):
            # Generate a hash for the item
            hash_val = int(hashlib.sha256((str(item) + str(i)).encode()).hexdigest(), 16)
            zeros = get_trailing_zeros(hash_val)
            if zeros > max_zeros[i]:
                max_zeros[i] = zeros

    # Estimate = 2^R where R is the average of max_zeros
    avg_max_zeros = sum(max_zeros) / num_hashes
    estimate = 2 ** avg_max_zeros
    return estimate

# Simulation
if __name__ == "__main__":
    # Simulate a stream of 1000 unique student IDs appearing multiple times
    stream = [f"STU{i%500}" for i in range(5000)] # 500 unique IDs

    print(f"Actual Unique IDs: 500")
    print(f"FM Estimate: {int(flajolet_martin(stream))}")
 Riverside: Standard practice often uses the median of averages of groups of hashes
# to improve accuracy, but this is a fundamental implementation.
