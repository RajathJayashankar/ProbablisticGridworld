import numpy as np
import random
from random import randint
from queue import PriorityQueue
from matplotlib import pyplot as plt

Node_type = {"isStart" : 1,"isEnd" : 2,"isNotBarrier" : 3, "isOpen": 4,"isBarrier" : 5, "isHilly" : 6, "isFlat" : 7, "isForest": 8}

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

matplotlibColorList = [CB91_Blue, CB91_Pink, CB91_Green, CB91_Amber, CB91_Purple, CB91_Violet]

def Manhattan_Distance(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return abs(x1 - x2) + abs(y1 - y2)

class Node:
  def __init__(self, row, col, total_rows):
    self.row = row
    self.col = col
    self.node_type = Node_type["isOpen"]
    self.terrainDiscovered = False
    self.neighbors = []
    self.total_rows = total_rows
    self.isPath = 0
    self.isTarget = False
    self.FNR = 0
    self.containTargetProb = 0
    self.findTargetProb = 0

  def get_pos(self):
    return self.row, self.col

  def is_barrier(self):
    return self.node_type == Node_type['isBarrier']

  def is_start(self):
    return self.node_type == Node_type['isStart']

  def is_end(self):
    return self.node_type == Node_type['isEnd']

  def make_inferred(self):
    self.inferred == True

  def make_start(self):
    self.node_type = Node_type['isStart']

  def make_barrier(self):
    self.node_type = Node_type['isBarrier']
    self.containTargetProb = 0

  def make_end(self):
    self.node_type = Node_type['isEnd']

  def make_target(self):
    self.isTarget = True

  def update_neighbors(self, grid):
    self.neighbors = []
    if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier(): # DOWN
      self.neighbors.append(grid[self.row + 1][self.col])

    if self.row > 0 and not grid[self.row - 1][self.col].is_barrier(): # UP
      self.neighbors.append(grid[self.row - 1][self.col])

    if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier(): # RIGHT
      self.neighbors.append(grid[self.row][self.col + 1])

    if self.col > 0 and not grid[self.row][self.col - 1].is_barrier(): # LEFT
      self.neighbors.append(grid[self.row][self.col - 1])

  def update_neighbors_barriers(self, grid, grid_original):
    if self.row < self.total_rows - 1 and grid_original[self.row + 1][self.col].is_barrier(): # DOWN
      grid[self.row + 1][self.col].make_barrier()

    if self.row > 0 and grid_original[self.row - 1][self.col].is_barrier(): # UP
      grid[self.row - 1][self.col].make_barrier()

    if self.col < self.total_rows - 1 and grid_original[self.row][self.col + 1].is_barrier(): # RIGHT
      grid[self.row][self.col + 1].make_barrier()

    if self.col > 0 and grid_original[self.row][self.col - 1].is_barrier(): # LEFT
      grid[self.row][self.col - 1].make_barrier()

  

  def __lt__(self, other):
    return False

def print_path(grid):
  li = []
  for i in range(len(grid)):
    li.append([])
    for j in range(len(grid[0])):
      li[i].append(grid[i][j].isPath)
  plt.figure()
  plt.imshow(li, cmap = 'Pastel2_r')

def reconstruct_path(came_from, current, grid):
  path_list = []
  while current in came_from:
    path_list.append(current.get_pos())
    current.isPath = 1
    current = came_from[current]
  print_path(grid)

def AStarAlgo(grid, start, end):
  count = 0
  open_set = PriorityQueue()
  open_set.put((0, count, start))
  came_from = {}
  g_score = {node: float("inf") for row in grid for node in row}
  g_score[start] = 0
  f_score = {node: float("inf") for row in grid for node in row}
  f_score[start] = Manhattan_Distance(start.get_pos(), end.get_pos())

  open_set_hash = {start}

  while not open_set.empty():
    current = open_set.get()[2]
    open_set_hash.remove(current)

    if current.get_pos() == end.get_pos():
      reconstruct_path(came_from, end, grid)
      return True

    current.update_neighbors(grid)
    for neighbor in current.neighbors:
      temp_g_score = g_score[current] + 1

      if temp_g_score < g_score[neighbor]:
        came_from[neighbor] = current
        g_score[neighbor] = temp_g_score
        f_score[neighbor] = temp_g_score + Manhattan_Distance(neighbor.get_pos(), end.get_pos())
        if neighbor not in open_set_hash:
          count += 1
          open_set.put((f_score[neighbor], count, neighbor))
          open_set_hash.add(neighbor)


  return False

def make_grid_density(rows, probability):
  grid = []
  for i in range(rows):
    grid.append([])
    for j in range(rows):
      node = Node(i, j, rows)
      node.containTargetProb = 1/(rows*rows)
      n = randint(0,100)
      if n < probability: node.make_barrier()
      grid[i].append(node)	

  return grid

def make_grid(rows):
  grid = []
  for i in range(rows):
    grid.append([])
    for j in range(rows):
      node = Node(i, j, rows)
      node.containTargetProb = 1/(rows*rows)
      grid[i].append(node)    
  return grid

def updateProbabilityBL(grid, x, y):
  Pbl = grid[x][y].containTargetProb
  for i in range(len(grid)):
      for j in range(len(grid[0])):
          if i == x and j == y:
            continue
          grid[i][j].containTargetProb = grid[x][y].containTargetProb/(1 - Pbl)

def reconstruct_path_repeated(grid, came_from, current, grid_orignal, path_traversed, came_from_2, metrics):
  path = []
  path.insert(0, current)
  while current in came_from:
    current = came_from[current]
    path.insert(0, current)

  unblocked_path = []

  for k in range(len(path)):
    metrics['movement'] += 1
    path[k].update_neighbors_barriers(grid, grid_orignal)
    if(grid_orignal[path[k].row][path[k].col].is_barrier() == False):
      unblocked_path.append(path[k])
    
      if (path[k] in came_from_2):
        indexSplice = came_from_2.index(path[k])
        came_from_2 = came_from_2[:indexSplice + 1]
      else:
        came_from_2.append(path[k])    
    else:
      metrics['movement'] += 1
      updateProbabilityBL(grid, path[k].row, path[k].col)
      path[k].make_barrier()      
      return unblocked_path, came_from_2, metrics
      
  return unblocked_path, came_from_2, metrics

def RepeatedAStar(grid, start, end, grid_orignal, path_traversed, came_from_2, metrics):
  count = 0
  open_set = PriorityQueue()
  open_set.put((0, count, start))
  came_from = {}

  g_score = {node: float("inf") for row in grid for node in row}
  g_score[start] = 0
  f_score = {node: float("inf") for row in grid for node in row}
  f_score[start] = Manhattan_Distance(start.get_pos(), end.get_pos())

  open_set_hash = {start}

  while not open_set.empty():
    current = open_set.get()[2]
    open_set_hash.remove(current)

    if current.get_pos() == end.get_pos():
      return reconstruct_path_repeated(grid, came_from, end, grid_orignal, path_traversed, came_from_2, metrics)

    current.update_neighbors(grid)
    for neighbor in current.neighbors:
      temp_g_score = g_score[current] + 1	

      if temp_g_score < g_score[neighbor]:
        came_from[neighbor] = current				
        g_score[neighbor] = temp_g_score
        f_score[neighbor] = temp_g_score + Manhattan_Distance (neighbor.get_pos(), end.get_pos())


        if neighbor not in open_set_hash:
          count += 1
          open_set.put((f_score[neighbor], count, neighbor))
          open_set_hash.add(neighbor)
  return path_traversed, came_from_2, metrics

def print_grid(grid):
  li = []
  for i in range(len(grid)):
    li.append([])
    for j in range(len(grid[0])):
      li[i].append(grid[i][j].node_type)
  plt.figure()
  plt.imshow(li, cmap = 'Pastel1_r')

def print_grid_path(grid):
  li = []
  for i in range(len(grid)):
    li.append([])
    for j in range(len(grid[0])):
      if(grid[i][j].isPath == 1):
         li[i].append(99)
      elif(i == y and y == j):
        li[i].append(99)
      else:
        li[i].append(grid[i][j].node_type)
  plt.figure()
  plt.imshow(li, cmap = 'Pastel1_r')

def print_grid_probabilities(grid):
  li = []
  for i in range(len(grid)):
    li.append([])
    for j in range(len(grid[0])):
      li[i].append(grid[i][j].containTargetProb)
  plt.figure()
  plt.imshow(li, cmap = 'viridis')

def make_grid_terrains(grid):
  for i in range(len(grid)):
    for j in range(len(grid[0])):
      p = np.random.rand()
      if(not grid[i][j].is_barrier()):
        if p <= 0.33:
          grid[i][j].node_type = Node_type['isHilly']
          grid[i][j].FNR = 0.5
        elif p > 0.33 and p <= 0.66:
          grid[i][j].node_type = Node_type['isFlat']
          grid[i][j].FNR = 0.2
        else:
          grid[i][j].node_type = Node_type['isForest']
          grid[i][j].FNR = 0.8
  return grid



#Creates a random target Location which is not blocked
def get_target(grid, rows):
  x = np.random.randint(rows)
  y = np.random.randint(rows)
  if grid[x][y].is_barrier():
    return get_target(grid, rows)
  else:
    return grid[x][y]

def getPresumedTargetLocation(grid, start):        
  arr = []
  maxVal = grid[0][0].containTargetProb # Take random maxVal as first value
  for i in range(len(grid)):
      for j in range(len(grid[0])):
          if grid[i][j].containTargetProb > maxVal and not grid[i][j].is_barrier() : # If current is greater than maxVal
            maxVal = grid[i][j].containTargetProb # Update maxVal
            arr = [] # Reset as new max value for prob has been found
            arr.append((grid[i][j], Manhattan_Distance(start.get_pos(), grid[i][j].get_pos())))
          elif grid[i][j].containTargetProb == maxVal: # Append all nodes that have prob same as max value
            arr.append((grid[i][j], Manhattan_Distance(start.get_pos(), grid[i][j].get_pos()))) 
  if  len(arr) == 1: # If only one max, return
    return arr[0][0]
  else:     
    arr = sorted(arr, key = lambda x : x[1])  # Sort based on 'distance'
    idx = 0
    liNo = []
    largest_val = arr[0][1]
    while arr[idx][1] == largest_val:
      liNo.append(arr[idx][0])
      idx += 1
    if len(liNo) > 1: # If still clash exists, return random node among these
      target_node = random.choice(liNo)
      return target_node
    else:
      return liNo[0]

def checkForTarget(target, currentNode):
  if(target.get_pos() != currentNode.get_pos()):
    return False
  FN = currentNode.FNR
  return random.choices([True, False], [1-FN, FN], k=1)[0]

def getFNR(node):
  if node.terrainDiscovered:
    return node.FNR
  else:
    (0.7/3)*sum([(1- key) for key in [0.8, 0.5, 0.2]])



def getCloseProb(grid, currTar, minVar, maxVar):
  li = []
  var = np.random(minVar, maxVar) # A min and max variance is set to find prob in this width, can be 0 too
  for i in range(len(grid)):
    for j in range(len(grid[0])):
      if currTar.findTargetProb - grid[i][j].findTargetProb > var:
        li.append(grid[i][j]) # Creates a list of all close by nodes in terms of prob of target
  return li

def updateHeuristics(grid, minVar, maxVar, currTar): #Updates utility for list of all cells with close prob
  node_list = getCloseProb(grid, currTar, minVar, maxVar)
  if Agent_Type == Agents.Agent8:
    heuristi_score = {}
    for node in node_list:
      sc = 0
      n = 0
      x, y = node.get_pos()
      for i in range(-2,2):
        for j in range(-2,2):
          x_, y_ = x + i, y + j
          if (x_, y_) in node_list:
            denom = (1/node_list[(x_,y_)][-1]) #Penalizing based on distance
            sc += node_list[(x_,y_)][0]*denom 
            n += 1
      cost = round(sc/n, 4) #Taking average of utiliy across the node neghbours to consider adjacent cells with high P_find
      heuristi_score[(x, y)] = sc # Heuristics are updated for corresposding x,y of list node


def updateProbabilities(grid, end, grid_original):
  x, y = end.get_pos()
  old_Pxy = end.containTargetProb
  FNR = grid_original[x][y].FNR

  new_Pxy = (FNR * old_Pxy) / (1 - (1 - FNR)*old_Pxy)
  end.containTargetProb = new_Pxy # Probabilty updation as in Question 2
  end.findTargetProb = (1 - FNR)*new_Pij # Probabilty updation as in Question 3

  for i in range(len(grid)):
      for j in range(len(grid[0])):
          if i==x and j==y:
              continue
          old_Pij = grid[i][j].containTargetProb
          new_Pij = old_Pij / (1 - (1 - FNR)*old_Pxy) # Probabilty updation as in Question 2
          grid[i][j].containTargetProb = new_Pij
          grid[i][j].findTargetProb = (1 - getFNR(grid[i][j]))*new_Pij # Probabilty updation as in Question 3

def printProbList(grid):
  for i in range(len(grid)):
    li = []
    for j in range(len(grid)):
      li.append(round(grid[i][j].containTargetProb, 4))
    print(li)

def main(rows):
  ROWS = rows
  
  pathExists = False

  grid_original = make_grid_density(ROWS, 30)
  start_orig = grid_original[0][0]
  start_orig.make_start()
  target = get_target(grid_original, rows)
  target.make_target()
  print_grid(grid_original)
  pathExists = AStarAlgo(grid_original, start_orig, target)
  grid_original = make_grid_terrains(grid_original)
  print_grid(grid_original)

  while not pathExists :    
    grid_original = make_grid_density(ROWS, 30)
    start_orig = grid_original[0][0]
    start_orig.make_start()
    target = get_target(grid_original, rows)
    target.make_target()
    print_grid(grid_original)
    pathExists = AStarAlgo(grid_original, start_orig, target)

  grid = make_grid(ROWS)  
  start = grid[0][0]
  print_grid(grid)
  printProbList(grid)
  metrics = {
      "movement" : 0,
      "examinations": 0,
      "runtime": 0
  }

  iterAgent6 = 0 
  foundAgent = False
  path_traversed = came_from = []
  end = getPresumedTargetLocation(grid, start)
  print(end.get_pos())
  while not foundAgent:
    while(True):
      path_traversed, came_from, metrics = RepeatedAStar(grid, start, end, grid_original, path_traversed, came_from, metrics) 
      iterAgent6 += 1

      if(path_traversed == [] or path_traversed == False):
        print("No Path")
        break	   

      if (path_traversed[-1] == end ):
        break

      else:
        start = path_traversed[-1]
 
    foundAgent = checkForTarget(target, end)
    if foundAgent or iterAgent6 > 1000:
      print("Found Target")
      break      
    else:
      updateProbabilities(grid, end, grid_original)
      start = end
      end = getPresumedTargetLocation(grid, start)

    printProbList(grid)
    print_grid_probabilities(grid)

  return grid, metrics

gr, mt = main(25)