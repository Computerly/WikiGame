#from sematch.semantic.similarity import WordNetSimilarity
import requests
import time
from urllib.parse import quote
import spacy

# To hide 0 vectors warning
import warnings
warnings.filterwarnings("ignore", message=r"\[W008\]", category=UserWarning)

class Node:
	def __init__(self, name, parent = None, cost = 0.0, heuristic = 0.0, token = None):
		self.name = name
		self.parent = parent
		self.cost = cost
		self.heuristic = heuristic
		self.token = token

	def __eq__(self, other) -> bool:
		return self.name.upper() == other.name.upper()
	
	def __hash__(self) -> int:
		return hash(self.name.upper())

	def __str__(self) -> str:
		return self.name


class WikiGame:
	def __init__(self, cost = 0.03):
		self.start = None
		self.goal = None
		self.costConstant = cost
		self.stats = dict()
		self.nlp = spacy.load("en_core_web_lg", exclude=["tagger", "parser", "senter", "attribute_ruler", "lemmatizer", "ner"])

	def getLinks(self, node: Node):
		# Send a GET request to the Wikipedia API to get the list of links on the page
		response = requests.get(f"https://en.wikipedia.org/w/api.php?action=parse&format=json&prop=links&page={quote(node.name)}")

		# Check status
		if response.status_code != 200:
			raise Exception(f"Failed to get page: '{node}'")

		# Parse the response and extract the list of links from the JSON data
		data = response.json()
		links = data["parse"]["links"]

		# Only use namespace = 0 and ensure the page exists
		links = filter(lambda l: True if l["ns"] in [0] and "exists" in l else False, links)

		# Remove unnecessary information
		links = list(map(lambda l: l["*"], links))

		# Return the list of links
		return links

	def getNeighbors(self, node: Node):
		nodes = []
		
		# Get all links on the page
		links = self.getLinks(node)
		
		# Preprocess each link and create a node
		for idx, doc in enumerate(self.nlp.pipe(links)):
			nodes.append(Node(links[idx], node, node.cost, 0.0, doc))
		#nodes = [Node(l, node, node.cost, None, 0.0, next) for l in self.nlp.pipe(links) ]
		return nodes
	'''
	Computes a heuristic for a node
	Returns a value between 0 and 1
	'''	
	def spacyHeuristic(self, current: Node, goal: Node):
		sim = current.token.similarity(goal.token) # ranges -1 to 1
		return 1 - ((sim + 1) / 2)
		

	def setStartName(self, name: str) -> bool:
		self.start = Node(name)
		# Check if valid
		return self.is_valid(self.start)


	def setGoalName(self, name: str) -> bool:
		self.goal = Node(name)
		# Check if valid
		isValid = self.is_valid(self.goal)
		if isValid:
			# Set token
			self.goal.token = self.nlp(self.goal.name)
		return isValid

	def is_valid(self, node: Node) -> bool:
		response = requests.get(f'https://en.wikipedia.org/w/api.php?action=parse&format=json&prop=links&page={quote(node.name)}&redirects&pllimit=1')

		if response.status_code != 200:
			return False

		data = response.json()

		if "error" in data:
			return False
		
		toPage = node.name
		redirects = data.get("parse", {}).get("redirects", {})
		if redirects != []:
			toPage = redirects[0].get("to", node.name)
			print(f"\tNotice! Redirecting from '{node.name}' to '{toPage}'")
		node.name = toPage

		# # sometime a wikipedia page is just a redirect page
		# if toPage != node.name
		# 	fromPage = data.get()["parse"]["redirects"][0]["from"]
		# 	toPage = data["parse"]["redirects"][0]["to"]
			
		# 	# Set node's name to redirect page
		# 	node.name = toPage

		# 	# Inform user
		# 	return True


		return True

	def getStats(self):
		return self.stats
	
	def solve(self, h = None, cost = 0.03):
		startTime = time.time()
		self.costConstant = cost
		
		if h == None:
			h = self.spacyHeuristic

		if self.start is None or self.goal is None:
			print(f"Invalid initialization. Start or goal not set")
			return
		# A priority queue. This will store nodes that have been visited and their associated costs.
		pq = [self.start]

		# Set of visited nodes.
		visited = {self.start}

		# While the frontier is not empty
		while pq:
			# Get the node with the lowest cost
			node = min(pq, key = lambda n:n.heuristic + n.cost) # faster than an actual priority queue
			pq.remove(node)
			# print(f"Exploring '{node}' f(n) = {node.cost} + {node.heuristic} = {node.cost + node.heuristic}")
			# If the node is the goal node:
			if node == self.goal:
				endTime = time.time()
				# Backtrack from the node to the starting node, adding each node to the optimal path list
				path = []
				while True:
					path.insert(0, node)

					if node.parent is None:
						self.stats["explored"] = len(visited)
						self.stats["time"] = (endTime - startTime)
						return path # Return the optimal path
					node = node.parent
			# Get all the neighbors of the node.
			neighbors = self.getNeighbors(node)
			
			for node in neighbors:
				if node not in visited:
					# Get the node's heuristic
					node.heuristic = h(node, self.goal)
					# Calculate the node's cost
					node.cost = node.cost + self.costConstant
					# Add the node to the frontier since it has not been explored.
					pq.append(node)
					# Add the node to visited.					
					visited.add(node)

		return [] # failed to find a path

if __name__ == "__main__":
	
	startName = "Canada"
	goalName = "Cat"
	
	game = WikiGame()

	while not game.setStartName(input("Enter First page name: ")):
		print("Invalid page!")

	while not game.setGoalName(input("Enter Goal page name: ")):
		print("Invalid page!")
	
	path = game.solve(cost=0.3)
	
	# Get game statisitics
	stats = game.getStats()
	
	# Print path
	if len(path):
		for i in range(len(path)-1):
			print(f"{path[i]} ->", end = " ")
		print(path[len(path) - 1])
	
		print(f"Degree of seperation: {len(path)-1}")
		print(f"Explored {stats['explored']} pages")
		print(f"Completed in {stats['time']} seconds")
	else: 
		print("Error no path found")
	

