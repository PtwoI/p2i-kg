"""Exercise the compact RL structures and their P2I bindings."""
import pytest


def test_incremental_fragments_preserve_user_curation(tmp_path):
    import json
    from pathlib import Path
    from p2i_kg.schema import KnowledgeGraph
    from p2i_kg.store import GraphStore, seed_graph

    root = Path(__file__).parents[1]
    fragments = [json.loads((root / 'docs' / name).read_text()) for name in
        ('rl-foundations.json', 'rl-planning.json', 'rl-dreamer-generations.json',
         'rl-skills.json', 'rl-classics.json', 'rl-lineage.json')]
    added_ids = {node['id'] for f in fragments for node in f['nodes']}
    data = seed_graph().model_dump(mode='json')
    data['nodes'] = [n for n in data['nodes'] if n['id'] not in added_ids]
    data['edges'] = [e for e in data['edges'] if e['source'] not in added_ids
                     and e['target'] not in added_ids and not (set(e['evidence']) & added_ids)]
    data['nodes'][0]['summary'] = 'Existing user curation survives.'
    store = GraphStore(tmp_path / 'catalog.sqlite3')
    store.replace(KnowledgeGraph.model_validate(data))
    for fragment in fragments:
        report = store.extend(fragment)
        assert report['added_nodes'] == len(fragment['nodes'])
        assert report['added_edges'] == len(fragment['edges'])
        assert store.extend(fragment)['added_nodes'] == 0
    assert store.load().nodes[0].summary == 'Existing user curation survives.'


@pytest.mark.parametrize(('name', 'target'), [
    ('dqn_rl', 'q_head'), ('ppo_rl', 'policy_head'),
    ('dreamer_rl', 'value_head'), ('diayn_rl', 'discriminator.2'),
    ('sac_rl', 'actor.2'), ('muzero_rl', 'value_head'),
    ('decision_transformer_rl', 'action_head'), ('dreamerv2_rl', 'value_head'),
    ('dreamerv3_rl', 'value_head'), ('dreamer4_rl', 'value_head'),
    ('dads_rl', 'policy.2'), ('aps_rl', 'actor.2'), ('option_critic_rl', 'option_values'),
    ('double_dqn_rl', 'q_head'), ('dueling_dqn_rl', 'value_stream'), ('rainbow_rl', 'value_stream'),
    ('reinforce_rl', 'policy_head'), ('actor_critic_rl', 'value_head'), ('a3c_rl', 'value_head'),
    ('ddpg_rl', 'critic_one.2'), ('td3_rl', 'critic_two.2'), ('alphazero_rl', 'policy_head'),
    ('planet_rl', 'reward_head'), ('world_models_rl', 'controller')])
def test_trace_edit_restore_backward(name, target):
    torch = pytest.importorskip('torch')
    p2i = pytest.importorskip('p2i')
    from p2i.checkpoint import checkpoint_bytes
    from p2i_kg.bindings import inspect_model
    from p2i_kg.runtime import make_model
    torch.set_num_threads(1)
    model, inputs = make_model(name)
    harness = p2i.Harness(model, example_inputs=inputs)
    assert harness.retrace(static=False).operations
    assert harness.apply({'type':'set_parameter', 'target':target,
                          'parameter':'bias', 'value':False}).committed
    fresh, _ = make_model(name)
    restored = p2i.Harness.load(checkpoint_bytes(harness), model=fresh)
    original, reloaded = harness.build()(*inputs), restored.build()(*inputs)
    for a, b in zip(original if isinstance(original, tuple) else (original,),
                    reloaded if isinstance(reloaded, tuple) else (reloaded,)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert restored.validate(backward=True).backward_valid
    assert inspect_model(name)['bindings']
