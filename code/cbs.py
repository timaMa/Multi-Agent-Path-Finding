import time as timer
import heapq
import random
import networkx as nx
import numpy as np
import math
import matplotlib.pyplot as plt
from single_agent_planner import compute_heuristics, a_star, \
get_location, get_sum_of_cost, construct_MDD_for_agent, reconstruct_MDD, updateMDD


def detect_collision(path1, path2):
	##############################
	# Task 3.1: Return the first collision that occurs between two robot paths (or None if there is no collision)
	#           There are two types of collisions: vertex collision and edge collision.
	#           A vertex collision occurs if both robots occupy the same location at the same timestep
	#           An edge collision occurs if the robots swap their location at the same timestep.
	#           You should use "get_location(path, t)" to get the location of a robot at time t.
	timestep = max(len(path1), len(path2))
	for t in range(timestep):
		loc1 = get_location(path1, t)
		loc2 = get_location(path2, t)
		if loc1 == loc2:
			return ([loc1], t)
		if t < timestep - 1:
			loc1_next = get_location(path1, t+1)
			loc2_next = get_location(path2, t+1)
			if loc1 == loc2_next and loc2 == loc1_next:
				return ([loc1, loc2], t+1)
	return None
	
def detect_collisions(paths):
	##############################
	# Task 3.1: Return a list of first collisions between all robot pairs.
	#           A collision can be represented as dictionary that contains the id of the two robots, the vertex or edge
	#           causing the collision, and the timestep at which the collision occurred.
	#           You should use your detect_collision function to find a collision between two robots.
	collisions = []
	num_of_agents = len(paths)
	for i in range(num_of_agents - 1):
		for j in range(i + 1, num_of_agents):
			collision_t = detect_collision(paths[i], paths[j])
			if collision_t is not None:
				collision = {'a1': i, 'a2': j, 'loc': collision_t[0], 'timestep': collision_t[1]}
				collisions.append(collision)
	return collisions
	
def standard_splitting(collision):
	##############################
	# Task 3.2: Return a list of (two) constraints to resolve the given collision
	#           Vertex collision: the first constraint prevents the first agent to be at the specified location at the
	#                            specified timestep, and the second constraint prevents the second agent to be at the
	#                            specified location at the specified timestep.
	#           Edge collision: the first constraint prevents the first agent to traverse the specified edge at the
	#                          specified timestep, and the second constraint prevents the second agent to traverse the
	#                          specified edge at the specified timestep
	constraints = []
	loc = collision['loc']
	timestep = collision['timestep']
	a1 = collision['a1']
	a2 = collision['a2']
	# Vertex collision
	if len(loc) == 1:
		constraints.append({'agent': a1, 'loc': loc, 'timestep': timestep, 'positive': False})
		constraints.append({'agent': a2, 'loc': loc, 'timestep': timestep, 'positive': False})
		return constraints
	if len(loc) == 2:
		reverse_loc = loc.copy()
		reverse_loc.reverse()
		constraints.append({'agent': a1, 'loc': loc, 'timestep': timestep, 'positive': False})
		constraints.append({'agent': a2, 'loc': reverse_loc, 'timestep': timestep, 'positive': False})
		return constraints
	
def disjoint_splitting(collision):
	##############################
	# Task 4.1: Return a list of (two) constraints to resolve the given collision
	#           Vertex collision: the first constraint enforces one agent to be at the specified location at the
	#                            specified timestep, and the second constraint prevents the same agent to be at the
	#                            same location at the timestep.
	#           Edge collision: the first constraint enforces one agent to traverse the specified edge at the
	#                          specified timestep, and the second constraint prevents the same agent to traverse the
	#                          specified edge at the specified timestep
	#           Choose the agent randomly

	constraints = []
	loc = collision['loc']
	timestep = collision['timestep']
	a1 = collision['a1']
	a2 = collision['a2']
	lucky_number = random.randint(0, 1)
	# Vertex collision
	if len(loc) == 1:
		if lucky_number == 0:
			constraints.append({'agent': a1, 'loc': loc, 'timestep': timestep, 'positive': True})
			constraints.append({'agent': a1, 'loc': loc, 'timestep': timestep, 'positive': False})
		else:
			constraints.append({'agent': a2, 'loc': loc, 'timestep': timestep, 'positive': True})
			constraints.append({'agent': a2, 'loc': loc, 'timestep': timestep, 'positive': False})
		return constraints
	if len(loc) == 2:
		reverse_loc = loc.copy()
		reverse_loc.reverse()
		if lucky_number == 0:
			constraints.append({'agent': a1, 'loc': loc, 'timestep': timestep, 'positive': True})
			constraints.append({'agent': a1, 'loc': loc, 'timestep': timestep, 'positive': False})
		else:
			constraints.append({'agent': a2, 'loc': reverse_loc, 'timestep': timestep, 'positive': True})
			constraints.append({'agent': a2, 'loc': reverse_loc, 'timestep': timestep, 'positive': False})
		return constraints
	
def paths_violate_constraint(constraint, paths):
	assert constraint['positive'] is True
	rst = []
	for i in range(len(paths)):
		if i == constraint['agent']:
			continue
		curr = get_location(paths[i], constraint['timestep'])
		prev = get_location(paths[i], constraint['timestep'] - 1)
		if len(constraint['loc']) == 1:  # vertex constraint
			if constraint['loc'][0] == curr:
				rst.append(i)
		else:  # edge constraint
			if constraint['loc'][0] == prev or constraint['loc'][1] == curr \
					or constraint['loc'] == [curr, prev]:
				rst.append(i)
	return rst


# Construct initial MDDs for all agents
def construct_MDD(my_map, num_of_agents,starts, goals, h_values, paths, constraints):
	MDD = []
	# Construct MDD for every agent
	for i in range(num_of_agents):
		MDD.append(construct_MDD_for_agent(my_map, i, starts[i], goals[i], h_values[i], len(paths[i]) - 1, constraints)) 
	return MDD

# Compute CG heuristic
def compute_CG(MDD, num_of_agents, starts):
	# Construct conflict graph according to cardinal conflict
	conflict_graph = construct_conflict_graph(num_of_agents, MDD, starts)
	# Compute CG heuristic from minimum vertex cover
	h_value = get_MVC(conflict_graph) # construct confict_graph
	# h_value = 0
	return h_value

def compute_DG(MDD, num_of_agents, starts, goals):
	# Construct confict graph according to cardinal conflict
	dependency_graph = construct_dependency_graph(num_of_agents, MDD, starts, goals)
	# Compute CG heuristic from minimum vertex cover
	h_value = get_MVC(dependency_graph) # construct confict_graph
	# h_value = 0
	return h_value

def compute_WDG(my_map, MDD, paths, constraints, num_of_agents, starts, goals):
	# Construct confict graph according to cardinal conflict
	dependency_graph = construct_dependency_graph(num_of_agents, MDD, starts, goals)
	# Compute weights and add them to dependency_graph
	weighted_graph = compute_weights(my_map, paths, constraints, num_of_agents, dependency_graph, starts, goals)
	# Compute WDG heuristic from edge-weighted minimum vertex cover
	h_value = compute_EWMVC(weighted_graph)
	return h_value

# Compute weights for each edge in the dependecy graph
def compute_weights(my_map, paths, constraints, num_of_agents, dependency_graph, starts, goals):
	G = dependency_graph.copy()
	# For every pair of agents
	for i in range(num_of_agents - 1):
		for j in range(i, num_of_agents):
			# If agent i and agent j have conflicts
			if (i, j) in G.edges:
				# Use CBS to get the cost of conflict-free paths between two agent
				constraints_ij = [constraint.copy() for constraint in constraints if constraint['agent']==i or constraint['agent']==j]
				for constraint in constraints_ij:
					if constraint['agent'] == i:
						constraint['agent'] = 0
					elif constraint['agent'] == j:
						constraint['agent'] = 1
				starts_2 = [starts[i], starts[j]]
				goals_2 = [goals[i], goals[j]]
				cbs = CBSSolver(my_map, starts_2, goals_2)
				cost, root_paths, root_constraints = cbs.find_solution(disjoint = False, heuristic = 'None', weight = True, constraints = constraints_ij) 
				weight = cost - len(paths[i]) - len(paths[j]) + 2
				G.add_edge(i, j, weight = weight)
	return G

# Compute the size of EWMVC of the dependency graph
# Use branch and bound
def compute_EWMVC(dependency_graph):
	G = dependency_graph.copy()
	# Divide dependency graph into multiple connected components
	connected_subgraphs = [G.subgraph(c) for c in nx.connected_components(G)]
	# Compute EWMVC
	EWMVC = 0
	for component in connected_subgraphs:
		EWMVC += compute_EWMVC_component(component)
	# print(EWMVC)
	return EWMVC

# Compute EWMVC for connected component
def compute_EWMVC_component(graph):
	G = graph.copy()
	num_of_nodes = nx.number_of_nodes(G)
	nodes = nx.nodes(G)
	possible_values = {}
	# Get possible values for each agent
	for node in nodes:
		possible_values[node] = []
		maximum = min([G.edges[edge]['weight'] for edge in G.edges(node)])
		for i in range(maximum + 1):
			possible_values[node].append(i)
	value_list = {}
	best = float('inf')
	best = bnb(possible_values, value_list, nodes, G, best)	
	return best

def bnb(possible_values, value_list, nodes, G, best):
	if len(value_list) == len(possible_values):
		cost = sum([value for _, value in value_list.items()])
		return cost
	else:
		value_list_copy = value_list.copy()
		unassigned_nodes = [node for node in nodes if node not in value_list_copy]
		node = unassigned_nodes[0]
		for value in possible_values[node]:
			if isViolated(value_list_copy, G, node, value):
				continue
			value_list_copy[node] = value
			cost = bnb(possible_values, value_list_copy, nodes, G, best)
			if cost < best:
				best = cost
		return best

def isViolated(value_list, G, node, value):
	for key, val in value_list.items():
		# If the node is connected to other nodes that is already assigned
		if (key, node) in G.edges():
			# Check whether the sum of two nodes 
			# are greater than the weight of the edge betwen them
			if val + value < G.edges[key, node]['weight']:
				return True
	return False

def construct_dependency_graph(num_of_agents, MDD, starts, goals):
	dependency_graph = nx.Graph()
	# Check whether agent i and agent j have cardinal conflicts
	for i in range(num_of_agents - 1):
		for j in range(i + 1, num_of_agents):
			# Merge two MDDs
			joint_MDD, max_level = merge_MDD(MDD[i], starts[i], goals[i], MDD[j], starts[j], goals[j])
			# If two agent are dependent
			if isDependent(joint_MDD, goals[i], goals[j], max_level) \
			or hasCardinal(MDD[i], starts[i], MDD[j], starts[j]):
				dependency_graph.add_nodes_from([i, j])
				dependency_graph.add_edge(i, j)
	return dependency_graph

# Merge two MDDs
def merge_MDD(MDD1, start1, goal1, MDD2, start2, goal2):
	# If depths of MDD1 and MDD2 are not the same
	len1 = len(reconstruct_MDD(MDD1, start1))
	len2 = len(reconstruct_MDD(MDD2, start2))
	MDD1_copy = MDD1.copy()
	MDD2_copy = MDD2.copy()
	if len1 > len2:
		edges = []
		for i in range(len2, len1):
			edges.append(((goal2, i-1), (goal2, i)))
		MDD2_copy.add_edges_from(edges)
	elif len1 < len2:
		edges = []
		for i in range(len1, len2):
			edges.append(((goal1, i-1), (goal1, i)))
		MDD1_copy.add_edges_from(edges)
	# Merge MDDs
	joint_MDD = {0:[(start1, start2)]}
	for i in range(max(len1, len2) - 1):
		joint_MDD[i+1] = []
		# For each pair at level i
		for pair in joint_MDD[i]:
			successor1 = [successor for successor, _ in list(MDD1_copy.successors((pair[0], i)))]
			successor2 = [successor for successor, _ in list(MDD2_copy.successors((pair[1], i)))]
			cross_product = [(x, y) for x in successor1 for y in successor2 if x != y]
			for new_pair in cross_product:
				if new_pair not in joint_MDD[i+1]:
					joint_MDD[i+1].append(new_pair)
		if len(joint_MDD[i+1]) == 0:
			return joint_MDD, max(len1, len2)-1
			
	return joint_MDD, max(len1, len2)-1

# Whether two agents are dependent
def isDependent(joint_MDD, goal1, goal2, max_level):
	# If the joint MDD has arrived at the max level
	if max_level in joint_MDD:
		if (goal1, goal2) in joint_MDD[max_level]:
			return False
	return True

# Construct confict graph according to cardinal conflict
def construct_conflict_graph(num_of_agents, MDD, starts):
	conflict_graph = nx.Graph()
	# Check whether agent i and agent j have cardinal conflicts
	for i in range(num_of_agents - 1):
		for j in range(i + 1, num_of_agents):
			# If agent i and agent j have cardinal conflicts
			if hasCardinal(MDD[i], starts[i], MDD[j], starts[j]):
				conflict_graph.add_nodes_from([i, j])
				conflict_graph.add_edge(i, j)
	return conflict_graph

# A conflict is cardinal 
# iff the contested vertex (or edge) is the only vertex (or edge) 
# at level t of the MDDs for both agents. 
def hasCardinal(MDD1, start1, MDD2, start2):
	# Reconstruct MDD according to timestep
	MDD1 = reconstruct_MDD(MDD1, start1)
	MDD2 = reconstruct_MDD(MDD2, start2)
	# Get the smaller cost of two MDD
	cost = min(len(MDD1), len(MDD2))
	for timestep in range(cost):
		# Cardinal vertex
		if len(MDD1[timestep]) == 1 and len(MDD2[timestep]) == 1 \
		and MDD1[timestep][0] == MDD2[timestep][0]:
			return True
		# Cardinal Edge
		if timestep < cost - 1:
			if len(MDD1[timestep]) == 1 and len(MDD2[timestep]) == 1 \
			and len(MDD1[timestep+1]) == 1 and len(MDD2[timestep+1]) == 1 \
			and MDD1[timestep][0] == MDD2[timestep+1][0] \
			and MDD1[timestep+1][0] == MDD2[timestep][0]:
				return True
	return False

# Compute minimum vertex cover
def get_MVC(G):
	upperbound = nx.number_of_nodes(G)
	C = []
	MVC = EMVC(G, upperbound, C)
	return MVC

# Recursive algorithm to find a lower bound of the size of minimum vertex cover
def EMVC(G, upperbound, C):
	if nx.is_empty(G):
		return len(C)
	# Compute clique-based lower bound
	cliques = get_disjoint_cliques(G)
	ClqLB = 0
	for clique in cliques:
		ClqLB += len(clique) - 1
	# Compute degree-based lower bound
	H = G.copy()
	num_of_edges = nx.number_of_edges(G)
	nodes = []
	degrees = []
	for degree in G.degree():
		nodes.append(degree[0])
		degrees.append(degree[1])
	DegLB = compute_DegLB(H, nodes, degrees, num_of_edges)
	# Compute MVC
	if len(C) + max(DegLB, ClqLB) >= upperbound:
		return upperbound
	
	# Select a vertex v from V with the maximum degree
	largest_index = np.argmax(degrees)
	vertex = nodes[largest_index]
	# Get the neighbours of the vertex with the maximum degree
	neighbors = [n for n in G.neighbors(vertex)]
	A = G.copy()
	A.remove_nodes_from(neighbors)
	A.remove_node(vertex)
	B = G.copy()
	B.remove_node(vertex)
	c1 = EMVC(A, upperbound, C + neighbors)
	c2 = EMVC(B, min(upperbound, c1), C + [vertex])
	return min(c1, c2)

def compute_DegLB(H, nodes, degrees, num_of_edges):
	i = 0
	total_degrees = 0
	while total_degrees < num_of_edges:
		# Select the vertex with the largest degree
		largest_index = np.argmax(degrees)
		total_degrees += degrees[largest_index]
		H.remove_node(nodes[largest_index])
		degrees.remove(degrees[largest_index])
		nodes.remove(nodes[largest_index])
		i += 1
	num_of_edges_afterRemove = nx.number_of_edges(H)
	max_degree_afterRemove = max(degrees)
	DegLB = math.floor(i+num_of_edges_afterRemove/max_degree_afterRemove)
	return DegLB	

def get_disjoint_cliques(G):
	disjoint_cliques = []
	existing_nodes = []
	# Get all the maximal cliques
	cliques = list(nx.find_cliques(G))
	# Sort them by their sizes
	cliques.sort(key = len, reverse = True)
	for clique in cliques:
		if len(disjoint_cliques) == 0:
			disjoint_cliques.append(clique)
			# Mark nodes in the clique as existing nodes
			existing_nodes = existing_nodes + clique
		else:
			# If there are nodes already marked as existing nodes
			if len(set(clique).intersection(set(existing_nodes))) == 0:
				disjoint_cliques.append(clique)
				existing_nodes = existing_nodes + clique
	if nx.number_of_nodes(G) == len(existing_nodes):
		return disjoint_cliques
	else:
		# Nodes that are not in the existing nodes
		nodes = [node for node in nx.nodes(G) if node not in existing_nodes]
		# Subgraph that contains nodes aht are not in the existing ndoes
		subgraph = G.subgraph(nodes)
		disjoint_cliques = disjoint_cliques + get_disjoint_cliques(subgraph)
		return disjoint_cliques

def draw_graph(G):
	pos = nx.spring_layout(G)  # positions for all nodes
	# nodes
	nx.draw_networkx_nodes(G, pos, node_size=400)
	# edges
	nx.draw_networkx_edges(G, pos, edgelist=G.edges, width=6)
	# labels
	nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')
	labels = nx.get_edge_attributes(G,'weight')
	nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

	plt.axis('off')
	plt.show()

class CBSSolver(object):
	"""The high-level search of CBS."""

	def __init__(self, my_map, starts, goals):
		"""my_map   - list of lists specifying obstacle positions
		starts      - [(x1, y1), (x2, y2), ...] list of start locations
		goals       - [(x1, y1), (x2, y2), ...] list of goal locations
		"""

		self.my_map = my_map
		self.starts = starts
		self.goals = goals
		self.num_of_agents = len(goals)
		self.heuristic = 'None'

		self.num_of_generated = 0
		self.num_of_expanded = 0
		self.CPU_time = 0
		self.construct_MDD = 0
		self.update_MDD = 0
		self.open_list = []

		# compute heuristics for the low-level search
		self.heuristics = []
		for goal in self.goals:
			self.heuristics.append(compute_heuristics(my_map, goal))

	def push_node(self, node):
		heapq.heappush(self.open_list, (node['cost'], len(node['collisions']), self.num_of_generated, node))
		# print("Generate node {}".format(self.num_of_generated))
		self.num_of_generated += 1

	def pop_node(self):
		_, _, id, node = heapq.heappop(self.open_list)
		# print("Expand node {}".format(id))
		self.num_of_expanded += 1
		return node

	def find_solution(self, disjoint=True, heuristic='None', weight = False, constraints=[]):
		""" Finds paths for all agents from their start locations to their goal locations

		disjoint    - use disjoint splitting or not
		"""
		self.heuristic = heuristic
		self.start_time = timer.time()

		# Generate the root node
		# constraints   - list of constraints
		# paths         - list of paths, one for each agent
		#               [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
		# collisions     - list of collisions in paths
		root = {'cost': 0,
				'constraints': [],
				'paths': [],
				'collisions': [],
				'MDD': []}
		root['constraints'] = constraints.copy()
		for i in range(self.num_of_agents):  # Find initial path for each agent
			path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
						  i, root['constraints'])
			if path is None:
				raise BaseException('No solutions')
			root['paths'].append(path)

		
		
		root['cost'] = get_sum_of_cost(root['paths'])
		root['collisions'] = detect_collisions(root['paths'])

		# Construct initial MDD
		# Record the time for constructing new MDD
		if heuristic != 'None':
			start_construct = timer.time()
			MDD = construct_MDD(self.my_map, self.num_of_agents, self.starts, self.goals, self.heuristics, root['paths'], [])
			self.construct_MDD += timer.time() - start_construct
			h = 0
			if heuristic == 'CG':
				# Compute CG heuristic
				h= compute_CG(MDD, self.num_of_agents, self.starts)
				
			elif heuristic == 'DG':
				# Compute DG heuristic
				h = compute_DG(MDD, self.num_of_agents, self.starts, self.goals)
				
			elif heuristic == 'WDG':
				# Compute WDG heuristic
				h = compute_WDG(self.my_map, MDD, root['paths'],root['constraints'], self.num_of_agents, self.starts, self.goals)
	
			root['MDD'] = MDD
			
			# Store the MDD for each agent without any constraints
			MDD_all = []
			for i in range(self.num_of_agents):
				mdd_i = {}
				mdd_i[len(root['paths'][i])-1] = MDD[i].copy()
				MDD_all.append(mdd_i)

		self.push_node(root)
		##############################
		# Task 3.3: High-Level Search
		#           Repeat the following as long as the open list is not empty:
		#             1. Get the next node from the open list (you can use self.pop_node()
		#             2. If this node has no collision, return solution
		#             3. Otherwise, choose the first collision and convert to a list of constraints (using your
		#                standard_splitting function). Add a new child node to your open list for each constraint
		#           Ensure to create a copy of any objects that your child nodes might inherit
		while len(self.open_list) > 0:
			P = self.pop_node()
			if len(P['collisions']) == 0:
				if weight:
					cost = get_sum_of_cost(P['paths'])
					return cost, P['paths'], root['constraints']
				self.print_results(P)
				return P['paths']
			collision = P['collisions'][0]
			if disjoint:
				constraints = disjoint_splitting(collision)
			else:
				constraints = standard_splitting(collision)
			for constraint in constraints:
				isAdd = True
				Q = {}
				Q['constraints'] = P['constraints'] + [constraint]
				Q['paths'] = [path.copy() for path in P['paths']]
				Q['MDD'] = [MDD.copy() for MDD in P['MDD']]
				# Q['paths'] = P['paths'].copy()
				if constraint['positive'] == False:
					a = constraint['agent']
					path = a_star(self.my_map, self.starts[a], self.goals[a], self.heuristics[a],
						  a, Q['constraints']) 
					if path is not None:
						Q['paths'][a] = path.copy()
						if heuristic != 'None':
							# Update MDD for agent a
							if len(P['paths'][a]) < len(path):
							# Construct new MDD with new depth for agent a
								mdd_temp = 0
								# If the MDD with new depth is already in the depth
								if (len(path) - 1) in MDD_all[a]: 
									
									mdd_temp = MDD_all[a][len(path)-1].copy()
								else:
									# Record the time for constructing new MDD
									start_construct = timer.time()
									# Construct MDD with new depth for agent a
									mdd_temp =  construct_MDD_for_agent(self.my_map, a, self.starts[a], self.goals[a],self.heuristics[a], len(path) - 1, [])
									self.construct_MDD += timer.time() - start_construct
									MDD_all[a][len(path)-1] = mdd_temp.copy()
								Q['MDD'][a] = mdd_temp.copy()
							# Record the time for updating MDD
							start_update = timer.time()
							Q['MDD'][a] = updateMDD(Q['MDD'][a], a, self.starts[a], self.goals[a], len(path) - 1, Q['constraints'])
							self.update_MDD += timer.time() - start_update
					else:
						isAdd = False
					
				# # if constraint['positive'] == True:
				# else:
				# 	rst = paths_violate_constraint(constraint, Q['paths'])
				# 	for i in rst: 
				# 		path = a_star(self.my_map, self.starts[i], self.goals[i], self.heuristics[i],
				# 				  i, Q['constraints'])
				# 		if path is not None:
				# 			Q['paths'][i] = path.copy()
				# 		else:
				# 			isAdd = False
				# 			break
				if isAdd:
					Q['collisions'] = detect_collisions(Q['paths'])
					h_value = 0
					if heuristic == 'CG':
						# Compute CG heuristic
						h_value = compute_CG(Q['MDD'], self.num_of_agents, self.starts)
					elif heuristic == 'DG':
						# Compute DG heuristic
						h_value = compute_DG(Q['MDD'], self.num_of_agents, self.starts, self.goals)
					elif heuristic == 'WDG':
						# Compute DG heuristic
						h_value = compute_WDG(self.my_map, Q['MDD'], Q['paths'],Q['constraints'], self.num_of_agents, self.starts, self.goals)
						
					Q['cost'] = get_sum_of_cost(Q['paths']) + h_value
					
					self.push_node(Q)
		
		return root['paths']


	def print_results(self, node):
		print("\n Found a solution! \n")
		CPU_time = timer.time() - self.start_time
		print("Use heuristic:    {}".format(self.heuristic))
		print("CPU time (s):    {:.2f}".format(CPU_time))
		print("Construct MDD time (s):    {:.2f}".format(self.construct_MDD))
		print("Update MDD time (s):    {:.2f}".format(self.update_MDD))
		print("Run time (s):    {:.2f}".format(CPU_time-self.construct_MDD - self.update_MDD))
		print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
		print("Expanded nodes:  {}".format(self.num_of_expanded))
		print("Generated nodes: {}".format(self.num_of_generated))
