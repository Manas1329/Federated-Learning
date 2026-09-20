import hashlib

class BloomFilter:
    def __init__(self, size=10000, hash_count=5):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [0] * size

    def _hashes(self, item):
        hashes = []
        for i in range(self.hash_count):
            # Create unique hash for each index
            hash_res = int(hashlib.md5((str(item) + str(i)).encode()).hexdigest(), 16)
            hashes.append(hash_res % self.size)
        return hashes

    def add(self, item):
        for h in self._hashes(item):
            self.bit_array[h] = 1

    def contains(self, item):
        for h in self._hashes(item):
            if self.bit_array[h] == 0:
                return False
        return True

# Simulation
if __name__ == "__main__":
    # Suppose these are active students from SIS
    active_students = ["STU1001", "STU1002", "STU1003"]
    bf = BloomFilter()

    for s in active_students:
        bf.add(s)

    # Testing
    test_ids = ["STU1001", "STU9999"]
    for tid in test_ids:
        if bf.contains(tid):
            print(f"ID {tid}: Found (Possibly Active)")
        else:
            print(f"ID {tid}: Not Found (Definitely Inactive)")
