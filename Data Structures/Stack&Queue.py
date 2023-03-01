from collections import deque

# Stacks and Queues are essentially served by collections.deque()
# Lists can also be used, however for append/pop ops only - they also can't provide optimal Dynamic Malloc

class Stack:
    
    def __init__(self):
        self.container = deque()
        
    def push(self, val):
        self.container.append(val)
        
    def pop(self):
        return self.container.pop()
    
    def peek(self):
        return self.container[-1]
    
    def is_empty(self):
        return len(self.container) == 0
    
    def size(self):
        return len(self.container)
    

class Queue:
    
    def __init__(self):
        self.buffer = deque()
        
    def enqueue(self, val):
        self.buffer.appendleft(val)
        
    def dequeue(self):
        return self.buffer.pop()
    
    
    def is_empty(self):
        return len(self.buffer) == 0
    
    def size(self):
        return len(self.buffer)