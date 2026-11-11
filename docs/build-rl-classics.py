"""Generate the sourced classic-RL catalog fragment (not a model trainer)."""
import json
from pathlib import Path

# id, label, paper title, year, URL, authors, runtime, scope, components, concept, parent
SPECS = [
 ('double_dqn','Double DQN','Deep Reinforcement Learning with Double Q-learning',2015,'https://arxiv.org/abs/1509.06461',['Hado van Hasselt','Arthur Guez','David Silver'],'double_dqn_rl','동일한 Q-network 추론 구조를 보여줍니다. Double DQN의 온라인·타깃 network를 이용한 target 계산과 학습은 포함하지 않습니다.',['rl_encoder','rl_q_head'],'rl_value_learning','dqn'),
 ('dueling_dqn','Dueling DQN','Dueling Network Architectures for Deep Reinforcement Learning',2015,'https://arxiv.org/abs/1511.06581',['Ziyu Wang','Tom Schaul','Matteo Hessel','Hado van Hasselt','Marc Lanctot','Nando de Freitas'],'dueling_dqn_rl','작은 공유 encoder에서 value와 advantage를 분리하고 평균을 뺀 후 Q값을 합칩니다. Atari CNN 및 학습은 포함하지 않습니다.',['rl_encoder','rl_value_stream','rl_advantage_stream'],'rl_value_learning','dqn'),
 ('rainbow','Rainbow','Rainbow: Combining Improvements in Deep Reinforcement Learning',2017,'https://arxiv.org/abs/1710.02298',['Matteo Hessel','Joseph Modayil','Hado van Hasselt','Tom Schaul','Georg Ostrovski','Will Dabney','Dan Horgan','Bilal Piot','Mohammad Azar','David Silver'],'rainbow_rl','두 갈래 dueling head가 행동별 5개 categorical atom logits를 출력합니다. NoisyNet, prioritized replay, n-step target, 학습과 실제 Atari encoder는 생략합니다.',['rl_encoder','rl_value_stream','rl_advantage_stream'],'rl_value_learning','dqn'),
 ('reinforce','REINFORCE','Simple statistical gradient-following algorithms for connectionist reinforcement learning',1992,'https://doi.org/10.1007/BF00992696',['Ronald J. Williams'],'reinforce_rl','작은 관측 encoder와 정책 logits를 실행합니다. 반환값 기반 정책 gradient loss, 환경 및 학습은 생략합니다.',['rl_encoder','rl_actor'],'rl_policy_gradient',None),
 ('actor_critic','Actor–Critic','Actor-Critic Algorithms',1999,'https://papers.nips.cc/paper_files/paper/1999/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html',['Vijay Konda','John Tsitsiklis'],'actor_critic_rl','공유 encoder 뒤 정책 및 가치 head가 있는 소형 신경망입니다. TD update 및 학습은 포함하지 않습니다.',['rl_encoder','rl_actor','rl_critic'],'rl_actor_critic',None),
 ('a3c','A3C','Asynchronous Methods for Deep Reinforcement Learning',2016,'https://arxiv.org/abs/1602.01783',['Volodymyr Mnih','Adria Puigdomenech Badia','Mehdi Mirza','Alex Graves','Timothy Lillicrap','Tim Harley','David Silver','Koray Kavukcuoglu'],'a3c_rl','단일 actor–critic worker의 정책·가치 head를 실행합니다. 다중 비동기 worker, 전역 파라미터와 학습은 생략합니다.',['rl_encoder','rl_actor','rl_critic'],'rl_actor_critic','actor_critic'),
 ('ddpg','DDPG','Continuous Control with Deep Reinforcement Learning',2015,'https://arxiv.org/abs/1509.02971',['Timothy Lillicrap','Jonathan Hunt','Alexander Pritzel','Nicolas Heess','Tom Erez','Yuval Tassa','David Silver','Daan Wierstra'],'ddpg_rl','소형 결정적 actor와 상태·행동 critic을 실행합니다. 타깃 network, replay, 노이즈 및 학습은 생략합니다.',['rl_actor','rl_q_head'],'rl_actor_critic',None),
 ('td3','TD3','Addressing Function Approximation Error in Actor-Critic Methods',2018,'https://proceedings.mlr.press/v80/fujimoto18a.html',['Scott Fujimoto','Herke van Hoof','David Meger'],'td3_rl','결정적 actor와 두 state-action critic을 실행합니다. 지연 업데이트, target smoothing, replay 및 학습은 생략합니다.',['rl_actor','rl_q_head'],'rl_actor_critic','ddpg'),
 ('alphazero','AlphaZero','Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm',2017,'https://arxiv.org/abs/1712.01815',['David Silver','Thomas Hubert','Julian Schrittwieser','Ioannis Antonoglou','Matthew Lai','Arthur Guez','Marc Lanctot','Laurent Sifre','Dharshan Kumaran','Thore Graepel','Timothy Lillicrap','Karen Simonyan','Demis Hassabis'],'alphazero_rl','작은 공유 trunk와 정책 및 가치 head를 실행합니다. MCTS, 자기 대국, 원논문 residual CNN 및 학습은 포함하지 않습니다.',['rl_encoder','rl_actor','rl_critic'],'rl_self_play',None),
 ('planet','PlaNet','Learning Latent Dynamics for Planning from Pixels',2018,'https://arxiv.org/abs/1811.04551',['Danijar Hafner','Timothy Lillicrap','Ian Fischer','Ruben Villegas','David Ha','Honglak Lee','James Davidson'],'planet_rl','작은 관측 인코더, 결정론적 GRU 전이, 복원과 보상 예측을 실행합니다. 확률적 RSSM, CEM 계획기, 영상 입력과 학습은 포함하지 않습니다.',['rl_encoder','rl_posterior','rl_dynamics','rl_reward','rl_decoder'],'rl_learned_planning',None),
 ('world_models','World Models','World Models',2018,'https://arxiv.org/abs/1803.10122',['David Ha','Jürgen Schmidhuber'],'world_models_rl','작은 encoder, 결정론적 순환 memory, controller와 decoder를 실행합니다. VAE sampling, MDN-RNN의 mixture density, 진화 탐색과 환경은 생략합니다.',['rl_encoder','rl_dynamics','rl_actor','rl_decoder'],'rl_latent_imagination',None),
]
nodes=[{'id':'concept:rl_policy_gradient','kind':'concept','label':'Policy gradient','summary':'정책 매개변수를 예상 return의 기울기로 최적화합니다.'},
 {'id':'concept:rl_self_play','kind':'concept','label':'Self-play and search','summary':'자기 대국과 탐색으로 정책·가치 학습 데이터를 만듭니다.'},
 {'id':'component:rl_value_stream','kind':'component','label':'Value stream','summary':'공유 특징에서 상태 가치 또는 가치 분포를 추정합니다.'},
 {'id':'component:rl_advantage_stream','kind':'component','label':'Advantage stream','summary':'공유 특징에서 행동별 상대적 이점을 추정합니다.'}]
edges=[]
def edge(src,rel,dst,paper,explanation,basis='documented'):
 edges.append({'source':src,'relation':rel,'target':dst,'basis':basis,'explanation':explanation,'evidence':[paper]})
for key,label,title,year,url,authors,runtime,note,components,concept,parent in SPECS:
 paper='paper:'+key;model='model:'+key
 nodes.append({'id':paper,'kind':'paper','label':title,'summary':label+'의 대표 논문입니다.','reference_year':year,'url':url,'authors':authors})
 nodes.append({'id':model,'kind':'model','label':label,'summary':label+' 강화학습 구조 및 알고리즘입니다.','reference_year':year,'family':'reinforcement-learning','runtime_id':runtime,'runtime_note':note})
 edge(model,'documented_in',paper,paper,'모델의 대표 논문입니다.')
 edge(model,'uses','concept:'+concept,paper,'해당 개념을 사용합니다.','structural')
 for component in components:edge(model,'has_component','component:'+component,paper,'실행 예제에 연결된 신경망 구성 요소입니다.','structural')
 edge(model,'trained_with','objective:rl_return',paper,'환경에서 장기 성과를 최적화합니다.','structural')
 edge(model,'applied_to','task:rl_control',paper,'순차 제어 또는 게임 의사결정에 사용합니다.','structural')
 if parent:edge(model,'extends','model:'+parent,paper,'명시된 기반 기법의 구조 또는 학습법을 확장합니다.','structural')
fragment={'schema_version':'0.2','nodes':nodes,'edges':edges}
Path('docs/rl-classics.json').write_text(json.dumps(fragment,ensure_ascii=False,indent=2)+'\n')
