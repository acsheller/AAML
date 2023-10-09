class SchedulerEnvironment:
    def __init__(self, nodes):
        self.nodes = nodes
        self.state = self.get_initial_state()

    def get_initial_state(self):
        state = []
        for node in self.nodes:
            state.append(node.available_cpu)
            state.append(node.available_memory)
        return state

    def get_possible_actions(self):
        return list(range(len(self.nodes)))

    def get_reward(self, action, pod):
        node = self.nodes[action]
        if node.available_cpu >= pod.cpu_request and node.available_memory >= pod.memory_request:
            return 1
        else:
            return -1

    def transition_state(self, action, pod):
        self.state[2*action] -= pod.cpu_request
        self.state[2*action + 1] -= pod.memory_request
        return self.state
