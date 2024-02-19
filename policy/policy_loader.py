from tf_agents.policies import policy_loader


def PolicyLoader(path):
    return lambda: policy_loader.load(path)
