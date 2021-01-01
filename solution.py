import matplotlib.pyplot as plt
import numpy as np
from queue import PriorityQueue
import pickle
import copy

class Node:
    def __init__(self, frequency = None, data=None):
        self.frequency = frequency
        self.left = None
        self.right = None
        self.data = data

    def __lt__(self, other):
        return ((self.frequency) < (other.frequency))
    
    def __str__(self):
        return str(self.data)
        
    def attachLeftNode(self, node):
        self.left = node
        
    def attachRightNode(self, node):
        self.right = node
        
    def navigateLeft(self):
        if self.left is None:
            self.left = Node()
        return self.left
        
    def navigateRight(self):
        if self.right is None:
            self.right = Node()
        return self.right
        
    def insert(self, data):
        # Compare the new value with the parent node
        if self.data:
            if data < self.data:
                if self.left is None:
                    self.left = Node(data)
                else:
                    self.left.insert(data)
            elif data > self.data:
                if self.right is None:
                    self.right = Node(data)
                else:
                    self.right.insert(data)
        else:
            self.data = data

class HuffmanEncoder:
    def findFrequency(self, inputData):
        return np.unique(inputData, return_counts=True)
    def buildBstFromFrequencyTable(self, freqTable):
        characters, frequencies = freqTable
        pq = PriorityQueue() # inbuilt module
        for index, frequency in enumerate(frequencies):
            pq.put(Node(frequency, characters[index]))
        root = None
        while not pq.empty():
            first_item = pq.get()
            if pq.empty():
                root = first_item
            else:
                second_item = pq.get()
                new_item = Node(first_item.frequency + second_item.frequency)
                new_item.attachRightNode(first_item)
                new_item.attachLeftNode(second_item)
                pq.put(new_item)
        return root
    def buildIndexingTableFromBst(self, node, indexingTable = None, value=''):
        if indexingTable is None:
            indexingTable = {}
        if node.data: indexingTable[node.data] = value
        else:
            indexingTable = self.buildIndexingTableFromBst(node.left, indexingTable, value+'0')
            indexingTable = self.buildIndexingTableFromBst(node.right, indexingTable, value+'1')
        return indexingTable
    def encodeWithIndexingTable(self, inputData, iTable):
        encodedText = []
        for char in inputData:
            digits = iTable[char]
            for digit in digits:
                encodedText.append(int(digit))
        return encodedText
    def encode(self, inputData):
        freqTable = self.findFrequency(inputData) # Tuple, Character frequencies
        bst = self.buildBstFromFrequencyTable(freqTable)
        indexingTable = self.buildIndexingTableFromBst(bst)
        return self.encodeWithIndexingTable(inputData, indexingTable), indexingTable
    def buildBstFromIndexingTable(self, encodings):
        bst = Node()
        for key in encodings:
            leaf = bst
            for character in encodings[int(key)]:
                if character == '0': leaf = leaf.navigateLeft()
                else: leaf = leaf.navigateRight()
            leaf.data = key
        return bst
    def decodeWithBst(self, inputData, bst):
        result = []
        current_node = bst
        for char in inputData:
            if char == 0:
                current_node = current_node.left
            else:
                current_node = current_node.right
            if current_node.data is not None:
                result.append(current_node.data)
                current_node = bst
        return result
    def decode(self, inputData, encodings):
        bst = self.buildBstFromIndexingTable(encodings)
        bst.display()
        outputData = self.decodeWithBst(inputData, bst)

class ImageEncoder:
    def __init__(self):
        self.huffmanEncoder = HuffmanEncoder()
    def color2gray(self, image):
        return np.round(np.mean(image, axis = 2), 0)
    def encode(self, imagePath):
        image = plt.imread(imagePath)
        image = self.color2gray(image)
        plt.imsave('grayImage.jpg', image)
        flattenedImage = image.ravel()
        
        binflattenedImage = copy.deepcopy(flattenedImage)
        converter = lambda t: format(int(t), '#010b')[2:]
        binflattenedImage = np.array([converter(xi) for xi in binflattenedImage])
        self.writeDataToFile(binflattenedImage, 'grayScaleOriginal.txt', True)
        
        encodedFlattenedImage, indexingTable = self.encodeFlattenedImage(flattenedImage)
        self.writeDataToFile(encodedFlattenedImage, 'encoded.txt', True)
        self.writePickle({'shape': image.shape, 'indexingTable': indexingTable}, 'indexingTable.pickle')
    def bitstring_to_bytes(self, s):
        ints = []
        for b in self.getbytes(iter(s)):
            ints.append(b)
        return ints
    def writePickle(self, data, pickleFileName):
        with open(pickleFileName, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    def readPickle(self, pickleFileName):
        with open(pickleFileName, 'rb') as handle:
            data = pickle.load(handle)
        return data
    def getbytes(self, bits):
        done = False
        while not done:
            byte = 0
            for _ in range(0, 8):
                try:
                    bit = next(bits)
                except StopIteration:
                    bit = 0
                    done = True
                byte = (byte << 1) | bit
            yield byte
    def writeDataToFile(self, data, fileName, toByte = False):
        if toByte is True:
            f = open(fileName, "wb")
            raw_data = bytes(data)
        else:
            f = open(fileName, "w")
            raw_data = data
        f.write(raw_data)
        f.close()
    def readDataFromFile(self, fileName):
        f = open(fileName, "rb")
        return list(f.read())
    def encodeFlattenedImage(self, flattenedImage):
        return self.huffmanEncoder.encode(flattenedImage)
    def decode(self, filePath):
        encodings = self.readPickle('indexingTable.pickle')
        bst = self.huffmanEncoder.buildBstFromIndexingTable(encodings['indexingTable'])
        image = self.huffmanEncoder.decodeWithBst(self.readDataFromFile(filePath), bst)
        image = np.array(image).reshape(encodings['shape'][0], encodings['shape'][1])
        
        plt.imsave('decodedImage.jpg', image)
    def decodeText(self, text, encodings):
        return self.huffmanEncoder.decode(text, encodings)

imageEncoder = ImageEncoder()
imageEncoder.encode('sunset.jpg')
imageEncoder.decode('encoded.txt')