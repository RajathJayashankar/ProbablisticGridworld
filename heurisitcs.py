
def Manhattan_Distance(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return abs(x1 - x2) + abs(y1 - y2)

def Weighted_Manhattan_Distance(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return 8*(abs(x1 - x2) + abs(y1 - y2))

def Euclidean_Distance(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return pow((pow((x1 - x2),2) + pow((y1 - y2), 2)), 0.5)

def Chebyshev_Distance(p1, p2):
	x1, y1 = p1
	x2, y2 = p2
	return max(abs(x1 - x2), abs(y1 - y2))

heuristic_functions = [Manhattan_Distance, Euclidean_Distance, Chebyshev_Distance]