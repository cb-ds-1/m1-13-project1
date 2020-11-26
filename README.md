## Dev notes

Different solutions were attempted with different levels of success:

- [x] play_game
- [x] q learn
- [x] mcts
- [x] last_attempt
- [x] radius

### play_game

this was my first attempt. I initially tried to create a move choosing policy where the bot would simiulate either a closest best bank approach or a random approach and choose the best.

I ended up discarding the random approach as it was always bad. The end result is simlations that generate 8.5 millions

### q learn

I attempted a q learn algorithm. The algorithm would have needed a lot of time and a better implementation to run well. results were poor at 200k.

### MCTS

I attempted a monte carlo tree search. I failed to create a proper game state tree. I spent a lot of time in rabit holes that I failed to get out of and ended up discarding this solution as I had to hand out something.

### last_attemp (not really the last attemps)

This was a mix of my failed MCTS and play_game approach. it was super slow and produced poor results. I droped it quickly

### Radius

This is an optimied version of playgame that plays from the end the game and looks within a radius of around 350 closest best banks. This is a really fast approach that could be used as the base of a really good algorithm. It completes under 10 seconds on a regular macbook.
