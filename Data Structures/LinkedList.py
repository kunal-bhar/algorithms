class Node:
    
    def __init__(self, data=None, next=None):
        self.data = data
        self.next = next

class LinkedList:
    
    def __init__(self):
        self.head = None
        
    def print(self):
        # itr through list and += into a string, edge case for empty
        if self.head is None:
            print('Empty List')
            return 
        
        itr = self.head
        llstr = ''
        while itr:
            llstr += str(itr.data) + '->'
            itr = itr.next
        print(llstr)
        
    def get_length(self):
        # itr through list and inc count 
        count = 0
        itr = self.head
        
        while itr:
            count += 1
            itr = itr.next
        return count 
    
    def insert_at_beginning(self, data):
        # init node w/ data and next= self.head (old), now update self.head w/ this node
        node= Node(data, self.head)
        self.head = node
        
    def insert_at_end(self, data):
        # itr through list while itr.next is not none, finally insert Node(data, None), direct insert @ head for empty
        if self.head is None:
            self.head = Node(data, None)
            return 
        
        itr= self.head
        while itr.next:
            itr = itr.next
        
        itr.next = Node(data, None)
        
    def insert_at(self, index, data):
        # validate idx, edge case for [0], itr through list till idx-1 and point itr.next to req node
        if index<0 or index>self.get_length():
            raise Exception('Invalid Index')
            
        if index == 0:
            self.insert_at_beginning(data)
            return
        
        count = 0
        itr = self.head
        while itr:
            if count == index - 1:
                node = Node(data, itr.next)
                itr.next = node
                break
            itr = itr.next
            count += 1
            
    def remove_at(self, index, data):
        # validate idx, edge case for [0], itr through list till idx-1 and set itr.next= itr.next.next
        if index<0 or index>self.get_length():
            raise Exception('Invalid Index')
            
        if index == 0:
            self.head = self.head.next
            return
        
        count = 0
        itr = self.head
        while itr:
            if count == index - 1:
                itr.next = itr.next.next
                break
            itr = itr.next
            count += 1
            
    def insert_values(self, data_list):
        # remove everything prev and itr through data_list w/ insert_at_end
        self.head = None
        for data in data_list:
            self.insert_at_end(data)      
                      