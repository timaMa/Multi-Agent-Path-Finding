import heapq
import networkx as nx

def move(loc, dir):
	directions = [(0, 0), (0, -1), (1, 0), (0, 1), (-1, 0)] 
	return loc[0] + directions[dir][0], loc[1] + directions[dir][1]


def get_sum_of_cost(paths):
	rst = 0
	for path in paths:
		rst += len(path) - 1
	return rst


def compute_heuristics(my_map, goal):
	# Use Dijkstra to build a shortest-path tree rooted at the goal location
	open_list = []
	closed_list = dict()
	root = {'loc': goal, 'cost': 0}
	heapq.heappush(open_list, (root['cost'], goal, root))
	closed_list[goal] = root
	while len(open_list) > 0:
		(cost, loc, curr) = heapq.heappop(open_list)
		for dir in range(5):
			child_loc = move(loc, dir)
			child_cost = cost + 1
			if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
			   or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
			   continue
			if my_map[child_loc[0]][child_loc[1]]:
				continue
			child = {'loc': child_loc, 'cost': child_cost}
			if child_loc in closed_list:
				existing_node = closed_list[child_loc]
				if existing_node['cost'] > child_cost:
					closed_list[child_loc] = child
					# open_list.delete((existing_node['cost'], existing_node['loc'], existing_node))
					heapq.heappush(open_list, (child_cost, child_loc, child))
			else:
				closed_list[child_loc] = child
				heapq.heappush(open_list, (child_cost, child_loc, child))

	# build the heuristics table
	h_values = dict()
	for loc, node in closed_list.items():
		h_values[loc] = node['cost']
	return h_values


def build_constraint_table(constraints, agent):
	##############################
	# Task 1.2/1.3: Return a table that constains the list of constraints of
	#               the given agent for each time step. The table can be used
	#               for a more efficient constraint violation check in the 
	#               is_constrained function.

	table = {}
	for constraint in constraints:
		if constraint['agent'] == agent:
			ts = constraint['timestep']
			if ts not in table:
				table[constraint['timestep']] = []
			table[ts].append(constraint)
		elif constraint['agent'] == 'goal':
			if 'goal' not in table:
				table['goal'] = []
			table['goal'].append(constraint)
		elif constraint['positive'] is True:
			constraint['agent'] = agent
			constraint['positive'] = False
			constraint['loc'].reverse()
			ts = constraint['timestep']
			if ts not in table:
				table[constraint['timestep']] = []
			table[ts].append(constraint)
	return table


def get_location(path, time):
	if time < 0:
		return path[0]
	elif time < len(path):
		return path[time]
	else:
		return path[-1]  # wait at the goal location


def get_path(goal_node):
	path = []
	curr = goal_node
	while curr is not None:
		path.append(curr['loc'])
		curr = curr['parent']
	path.reverse()
	return path


def is_constrained(curr_loc, next_loc, next_time, constraint_table):
	##############################
	# Task 1.2/1.3: Check if a move from curr_loc to next_loc at time step next_time violates
	#               any given constraint. For efficiency the cons traints are indexed in a constraint_table
	#               by time step, see build_constraint_table.
	if 'goal' in constraint_table:
		for constraint in constraint_table['goal']:
			if next_loc == constraint['loc'][0] and next_time >= constraint['timestep']:
				return True
	if next_time in constraint_table:
		constraints = constraint_table[next_time]
		for constraint in constraints:
			if constraint['positive'] == False:
				if len(constraint['loc']) == 1:
					if constraint['loc'][0] == next_loc:
						return True
				else:
					if constraint['loc'][0] == curr_loc and constraint['loc'][1] == next_loc:
						return True
			else:
				if len(constraint['loc']) == 1:
					if constraint['loc'][0] is not next_loc:
						return True
				else:
					if constraint['loc'][0] == curr_loc and constraint['loc'][1] is not next_loc:
						return True
	return False


def push_node(open_list, node):
	heapq.heappush(open_list, (node['g_val'] + node['h_val'], node['h_val'], node['loc'], node))


def pop_node(open_list):
	_, _, _, curr = heapq.heappop(open_list)
	return curr


def compare_nodes(n1, n2):
	"""Return true is n1 is better than n2."""
	return n1['g_val'] + n1['h_val'] < n2['g_val'] + n2['h_val']


def a_star(my_map, start_loc, goal_loc, h_values, agent, constraints):
	""" my_map      - binary obstacle map
		start_loc   - start position
		goal_loc    - goal position
		agent       - the agent that is being re-planned
		constraints - constraints defining where robot should or cannot go at each timestep
	"""

	##############################
	# Task 1.1: Extend the A* search to search in the space-time domain
	#           rather than space domain, only.

	# length_limit = 0
	# for l in my_map:
	#     length_limit = length_limit + l.count(False)

	open_list = []
	closed_list = dict()
	earliest_goal_timestep = 0
	h_value = h_values[start_loc]
	constraintTable = build_constraint_table(constraints, agent)
	
	# length_limit = length_limit + len(constraints)


	root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'timestep':0}
	push_node(open_list, root)
	closed_list[(root['loc'], root['timestep'])] = root
	path_length = 0
	while len(open_list) > 0:
		# if path_length > length_limit:
		#     print('yes')
		#     return None
		curr = pop_node(open_list)
		#############################
		# Task 1.4: Adjust the goal test condition to handle goal constraints
		# timesteps = [key for key in constraintTable.keys() if key != 'goal']
		# if curr['loc'] == goal_loc and (len(constraintTable) == 0 or curr['timestep']>max(timesteps)):
		#     return get_path(curr)
		goalConstraints = False
		for key in constraintTable.keys():
			if key != 'goal' and key > curr['timestep']:
				goal = {'loc': [goal_loc], 'agent': agent, 'timestep': key, 'positive': False}
				if goal in constraintTable[key]:
					goalConstraints = True
					break

		if curr['loc'] == goal_loc and not goalConstraints:
			return get_path(curr)

		for dir in range(5):
			child_loc = move(curr['loc'], dir)
			if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
			or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
				continue
			
			if my_map[child_loc[0]][child_loc[1]]:
				continue
			if is_constrained(curr['loc'], child_loc, curr['timestep'] + 1, constraintTable):
				continue
			
			child = {'loc': child_loc,
					'g_val': curr['g_val'] + 1,
					'h_val': h_values[child_loc],
					'parent': curr,
					'timestep': curr['timestep'] + 1}
			if (child['loc'], child['timestep']) in closed_list:
				existing_node = closed_list[(child['loc'], child['timestep'])]
				if compare_nodes(child, existing_node):
					closed_list[(child['loc'], child['timestep'])] = child
					push_node(open_list, child)
			else:
				closed_list[(child['loc'], child['timestep'])] = child
				push_node(open_list, child)
		path_length = path_length + 1

	return None  # Failed to find solutions

# Construct MDD for agent i
# Perform a BFS from the start location of agent down to depth c
# and only store the partial DAG which starts at start and ends at goal at depth
def construct_MDD_for_agent(my_map, agent, start_loc, goal_loc, h_values, cost, constraints):
	# MDD stores all paths that have the cost of c 
	MDD = nx.DiGraph()
	h_value = h_values[start_loc]
	open_list = []
	# build contraint table for agent
	constraintTable = build_constraint_table(constraints, agent)

	root = {'loc': start_loc, 'g_val': 0, 'h_val': h_value, 'parent': None, 'timestep':0}
	open_list.append(root)

	while len(open_list) > 0:
		# remove front 
		curr = open_list.pop(0)
		#############################
		# Task 1.4: Adjust the goal test condition to handle goal constraints
	
		# The current node is at depth c
		if curr['timestep'] == cost:
			# Check whether it is a goal node
			# If so, get the path and put it into MDD
			# If not, then do keep check the next node
			if curr['loc'] == goal_loc:
				path = get_path(curr)
				for i in range(len(path) - 1):
					# MDD.add_node(path[i])
					MDD.add_edge((path[i], i), (path[i+1], i+1))
			continue

		# Expand the current node
		for dir in range(5):
			# Choose a direction
			child_loc = move(curr['loc'], dir)
			# Check whether the child location is outside the map
			if child_loc[0] < 0 or child_loc[0] >= len(my_map) \
			or child_loc[1] < 0 or child_loc[1] >= len(my_map[0]):
				continue
			# Check whether the child location is an obstacle 
			if my_map[child_loc[0]][child_loc[1]]:
				continue
			# Check whether the child location violates any constraints
			if is_constrained(curr['loc'], child_loc, curr['timestep'] + 1, constraintTable):
				continue
			# Check whether the g_val + h_value is larger than cost
			if curr['g_val'] + h_values[child_loc] + 1 > cost:
				continue

			child = {'loc': child_loc,
					'g_val': curr['g_val'] + 1,
					'h_val': h_values[child_loc],
					'parent': curr,
					'timestep': curr['timestep'] + 1}
			open_list.append(child)
	return MDD  

# Reconstruct MDD according to timestep
def reconstruct_MDD(MDD, start_loc):
	# Construct MDD which indexes the locations by their time steps.
	new_MDD = {}
	# Append locations to their corresponding timesteps
	locations = nx.single_source_shortest_path_length(MDD, (start_loc, 0)) 
	for loc, depth in locations.items():
		if depth not in new_MDD:
			new_MDD[depth] = []
		new_MDD[depth].append(loc[0])
	return new_MDD

def updateMDD(MDD, agent, start_loc, goal_loc, cost, constraints):
	constraintTable = build_constraint_table(constraints, agent)
	MDD_copy = MDD.copy()
	recons_MDD = reconstruct_MDD(MDD, start_loc)
	# If the cost is same as the last time
	for timestep, locations in recons_MDD.items():	
		# If the current location is goal location
		# then stop search
		if locations[0] == goal_loc:	
			break
		else: 
			for curr_loc in locations:
				for next_loc in list(MDD_copy.successors((curr_loc,timestep))):
					if is_constrained(curr_loc, next_loc[0], timestep+1, constraintTable):
						MDD_copy.remove_edge((curr_loc, timestep), next_loc)
	# Remove nodes in the directed graph that do not have successors or predecessors
	deleted_nodes = []
	for node in nx.nodes(MDD_copy):
		if node == (start_loc, 0):
			continue
		elif node != (goal_loc, cost):
			if len(list(MDD_copy.predecessors(node))) == 0 \
			or len(list(MDD_copy.successors(node))) == 0:
				deleted_nodes.append(node)
	MDD_copy.remove_nodes_from(deleted_nodes)		
	return MDD_copy

