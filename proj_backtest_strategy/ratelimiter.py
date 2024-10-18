import time

class RateLimiter:
    def __init__(self, max_calls, period):
        self.calls = []
        self.max_calls = max_calls
        self.period = period

    def wait(self):
        now = time.monotonic()
        # Remove timestamps that are older than the rate limit period
        self.calls = [t for t in self.calls if now - t < self.period]
        if len(self.calls) >= self.max_calls:
            # Calculate how long to sleep
            sleep_time = self.period - (now - self.calls[0])
            if sleep_time > 0:
                print(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds.")
                time.sleep(sleep_time)
            now = time.monotonic()
            # Clean up old timestamps again after sleeping
            self.calls = [t for t in self.calls if now - t < self.period]
        self.calls.append(now)
