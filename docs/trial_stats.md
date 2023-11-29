ACDQN. Should be called MLP.
1. Steamed Glaze
hidden_layers=64,
gamma=0.9,
actor_learning_rate=1e-4,
critic_learning_rate=1e-4,
Results: Cumulative sum of rewareds is -75.871

a. quantum shark
hidden_layers=128
gamma=0.9
actor_learning_rate=1e-4
critic_learning_rate=1e-4
optimizer=2
Variance too high


b. Plastic Python -- Looking Good
hidden_layers=64,gamma=0.9
actor_learning_rate=1e-4
critic_learning_rate=1e-4
optimizer=0
Cumulative sum of rewareds is -48.016

c. 

2. 
hidden_layers=64,
gamma=0.9,
actor_learning_rate=1e-3,
critic_learning_rate=1e-4,
Results:  BAD ---  at about 80 steps  it  was assigning everything to node 0.


3. 
hidden_layers=64,
gamma=0.9,
actor_learning_rate=1e-4,
critic_learning_rate=1e-3,
Results: Cumulative sum of rewareds is -58.737
Results: Bad but not so bad.-- It finished but the 2nd epoch seems to have hurt it. The cluster variance is bad.

4. 
hidden_layers=64,gamma=0.9,
actor_learning_rate=1e-4,
critic_learning_rate=1e-4,
optimizer=1
Results: -97 CSR BAD

5. 
hidden_layers=64,gamma=0.9,
actor_learning_rate=1e-4,
critic_learning_rate=1e-4,
optimizer=1
Cumulative sum of rewareds is -46.6847.689
