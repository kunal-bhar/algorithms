# -*- coding: utf-8 -*-

# Dictionary as a Hash Map

# Steps:
# 1. The 'values' are stored in a array/list.
# 2. The array idx is determined by passing the 'key' through a hash fxn; the one implemented here is ASCII number sum // array size
# 3. set item: hashmap['a'] = 1
#    get item: print(hashmap['a'])
#    del item: del hashmap['a']


class HashMapWithoutChaining:
    
    def __init__(self):
        self.MAX = 10
        self.arr = [None for i in range(self.MAX)]
        
    def get_hash(self, key):
        hash = 0
        for char in key:
            hash += ord(char)
        return hash % self.MAX
   
    def __setitem__(self, key, val):
        h = self.get_hash(key)
        self.arr[h] = val
        
    def __getitem__(self, key):
        h = self.get_hash(key)
        return self.arr[h]
    
    def __delitem__(self, key):
        h = self.get_hash(key)
        self.arr[h] = None

    
class HashMap(): 
# Handles collision by storing multiple entries at the particular arr[idx], i.e in a list of lists

    def __init__(self):
        self.MAX = 10
        self.arr = [[] for i in range(self.MAX)]
        
    def get_hash(self, key):
        hash = 0
        for char in key:
            hash += ord(char)
        return hash % self.MAX
    
    def __setitem__(self, key, val):
        h = self.get_hash(key)
        exists = False
        for idx, element in enumerate(self.arr[h]):
            if len(element) == 2 and element[0] == key:
                self.arr[h][idx] = (key, val)
                exists = True
        if not exists:
            self.arr[h].append((key, val))
            
    def __getitem__(self, key):
        h = self.get_hash(key)
        for kv in self.arr[h]:
            if kv[0] == key:
                return kv[1]
    
    def __delitem__(self, key):
        h = self.get_hash(key)
        for idx, kv in enumerate(self.arr[h]):
            if kv[0] == key:
                del self.arr[h][idx]
    
    
