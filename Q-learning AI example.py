import numpy as np
from random import randint
import random

"""

s = state = état de l'agent

spt = state position au temps t
sot = state orientation au temps t
self.oreint = : 0 = right, 1 = up, 2 = left, 3 = down

at = action au temps t
Action: 0 = TURN_LEFT, 1 = TURN_RIGH, 2 = FORWARD, 3 = FORWARD_RIGHT, 4 = FORWARD_LEFT

sptp1 = state position au temps t+1
sotp1 = state orientation au temps t+1

atp1 = action au temps t+1

r = récompense 

"""

"""

Ce programme est un exemple d'apprentissage par renforcement via la méthode du Q-learning.

L'agent (l'IA) est ici représenté par un triangle qui peut excécuter 5 actions : tourner à gauche ou à droite,
avancer tout droit, en diagonale gauche ou en diagonale droite.
Son but est de maximiser son score.
Il est défini comme la somme de valeurs sur lesquelles l'agent est allé, divisé par le nombre de parties effectuées.

L'environnement est défini dans la méthode __init__ (ligne 58).
L'exécution des actions est géré par la méthode step (ligne 97).
La gestion des bonus est effectuée par la méthode reward (ligne 168).
L'affichage est effectué par la méthode show (ligne 237).
La prise de décision est faite par la fonction take_action (ligne 260).
La mémoire (l'expérience accumulé) de l'agent est gérée par une liste qu'on appelle Q-Table (ligne 279).
La "Q fonction" (ou équation de Bellman) permet à l'agent de mettre à jour sa perception de l'environnement(ligne 402).
Plus précisément, elle met à jour l'espérance du score que l'agent peut obtenir en faisant telle action dans tel état.
Notre Q-Table est un tableau à 3 dimensions défini selon 3 paramètres : La position de l'agent (100 positions), 
la direction vers laquelle il regarde (4 directions) et les actions qu'il peut effectuer (5 actions).
Pour faire une analogie, c'est comme si notre agent devait se repérer uniquement avec un GPS et sa proprioception
mais il n'a pas la capacité d'observer son environnement (pour l'instant ?).

"""

bonus = []
checkpoint_1 = 0
checkpoint_2 = 0
checkpoint_3 = 0
checkpoint_4 = 0
score = 0


class EnvGrid(object):
    def __init__(self):
        super(EnvGrid, self).__init__()

        self.grid = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                     [-1,  0,  0,  0,  0,  2,  0,  0,  0, -1],
                     [-1,  0,  0,  0,  0,  2,  0,  0,  0, -1],
                     [-1,  0,  0,  0,  0, -1, -1,  0,  0, -1],
                     [-1,  0,  0, -1, -1, -1, -1,  2,  2, -1],
                     [-1,  2,  2, -1, -1, -1, -1,  0,  0, -1],
                     [-1,  0,  0, -1, -1,  0,  0,  0,  0, -1],
                     [-1,  0,  0,  0,  2,  0,  0,  0,  0, -1],
                     [-1,  0,  0,  0,  2,  0,  0,  0,  0, -1],
                     [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]

        # Starting position
        self.y = 2
        self.x = 3
        self.orient = 0

        self.actions = [
            0,  # TURN_LEFT
            1,  # TURN_LEFT
            2,  # FORWARD_RIGHT
            3,  # FORWARD
            4,  # FORWARD_LEFT
        ]

    def reset(self):
        global checkpoint_1, checkpoint_2, checkpoint_3, checkpoint_4

        self.y = 2
        self.x = 3
        self.orient = 0
        checkpoint_1 = 20
        checkpoint_2 = 20  # fait réapparaître les bonus
        checkpoint_3 = 20
        checkpoint_4 = 20
        return(self.y*10+self.x+1), self.orient  # y*10 + x+1 : si y = 3 et x = 5, l'agent est dans l'état st = 36

    def step(self, action):
        """
            Action: 0 = TURN_LEFT, 1 = TURN_RIGH, 2 = FORWARD, 3 = FORWARD_RIGHT, 4 = FORWARD_LEFT
            Orientation: 0 = Right, 1 = Up, 2 = Left, 3 = Down
        """

        if self.actions[action] == 0:
            # print("TURN_LEFT")
            self.orient += 1
            if self.orient == 4:
                self.orient = 0

        elif self.actions[action] == 1:
            # print("TURN_RIGHT")
            self.orient -= 1
            if self.orient == -1:
                self.orient = 3

        elif self.actions[action] == 2:
            # print("FORWARD_RIGHT")
            if self.orient == 2:
                self.y += -1
                self.x += -1
            elif self.orient == 0:
                self.y += 1
                self.x += 1
            elif self.orient == 1:
                self.y += -1
                self.x += 1
            elif self.orient == 3:
                self.y += 1
                self.x += -1

        elif self.actions[action] == 3:
            # print("FORWARD")
            if self.orient == 2:
                self.x += -1
            elif self.orient == 0:
                self.x += 1
            elif self.orient == 1:
                self.y += -1
            elif self.orient == 3:
                self.y += 1

        elif self.actions[action] == 4:
            # print("FORWARD_LEFT")
            if self.orient == 2:
                self.y += 1
                self.x += -1
            elif self.orient == 0:
                self.y += -1
                self.x += 1
            elif self.orient == 1:
                self.y += -1
                self.x += -1
            elif self.orient == 3:
                self.y += 1
                self.x += 1

        return (self.y*10+self.x+1), self.orient, self.grid[self.y][self.x]  # return sptp1, sotp1, r

    def reward(self):
        global bonus, checkpoint_1, checkpoint_2, checkpoint_3, checkpoint_4

        checkpoint_1 += 1
        checkpoint_2 += 1  # Les checkpoints gagnent 1 pour chaque action de l'agent
        checkpoint_3 += 1
        checkpoint_4 += 1
        # print(bonus)
        # print(checkpoint_1, checkpoint_2, checkpoint_3, checkpoint_4)

        for _ in range(2):  # 2 fois pour actualiser sinon il a un cycle de retard

            if self.grid[self.y][self.x] == 2:  # Si l'agent passe sur un bonus on met dans une liste le bonus correspondant
                bonus.append(self.x)            # en x = 5, il y un bonus alors on ajoute 5 dans la liste bonus
                self.grid[self.y][self.x] = 0

            if 5 in bonus:  # Si 5 est dans la liste bonus, alors les cases [1][5] et [2][5] sont remplacées par des 0
                self.grid[1][5] = 0
                self.grid[2][5] = 0
            else:
                self.grid[1][5] = 2  # Sinon la case reste un 2
                self.grid[2][5] = 2
                checkpoint_1 = 0    # La valeur du checkpoint se réinitialise
            if checkpoint_1 >= 18:  # Si le chekpoint_1 dépasse 18, alors il se réinitialise à 0
                checkpoint_1 = 0
                bonus.remove(5)     # Puis on enlève la valeur 5 de la liste pour que le bonus puisse réappaître

            if 7 in bonus or 8 in bonus:  # même chose pour le 2e groupe de bonus
                self.grid[4][7] = 0
                self.grid[4][8] = 0
            else:
                self.grid[4][7] = 2
                self.grid[4][8] = 2
                checkpoint_2 = 0

            if checkpoint_2 >= 18:
                checkpoint_2 = 0
                if 7 in bonus:
                    bonus.remove(7)
                else:
                    bonus.remove(8)

            if 4 in bonus:
                self.grid[7][4] = 0
                self.grid[8][4] = 0
            else:
                self.grid[7][4] = 2
                self.grid[8][4] = 2
                checkpoint_3 = 0

            if checkpoint_3 >= 18:
                checkpoint_3 = 0
                bonus.remove(4)

            if 2 in bonus or 1 in bonus:
                self.grid[5][1] = 0
                self.grid[5][2] = 0
            else:
                self.grid[5][1] = 2
                self.grid[5][2] = 2
                checkpoint_4 = 0

            if checkpoint_4 >= 18:
                checkpoint_4 = 0
                if 2 in bonus:
                    bonus.remove(2)
                else:
                    bonus.remove(1)

    def show(self):  # Afficher le terrain

        y = 0
        for line in self.grid:
            x = 0
            for pt in line:
                if self.orient == 2:
                    print("{}\t".format(pt if y != self.y or x != self.x else "◄"), end="")
                elif self.orient == 0:
                    print("{}\t".format(pt if y != self.y or x != self.x else "►"), end="")
                elif self.orient == 1:
                    print("{}\t".format(pt if y != self.y or x != self.x else "▲"), end="")
                elif self.orient == 3:
                    print("{}\t".format(pt if y != self.y or x != self.x else "▼"), end="")
                x += 1
            y += 1
            print("")
        print("---------------------")

    def is_finished(self):  # fin de la partie quand l'agent va sur la valeur -1
        return self.grid[self.y][self.x] == -1


def take_action(spt, sot, Q, eps):

    if random.uniform(0, 1) < eps:  # Action aléatoire ?
        action = randint(0, 4)
    else:  # Action qui maximise l'espérance
        if np.argmax(Q[spt][sot]) == 0:  # Si elle n'est pas nulle, on favorise l'exploration dans un état peu connu
            action = randint(0, 4)
        else:
            action = np.argmax(Q[spt][sot])
    return action  # return at


if __name__ == '__main__':
    env = EnvGrid()

    rep = 10000  # nombre de parties d'entraînement (10 000 conseillé même si plus long)
    epsilon = 1  # probabilité que l'agent prenne une action aléatoirement

    Q = [[[0, 0, 0, 0, 0] for i in range(4)] for y in range(101)]
    # 100 listes de 4 listes de 5 valeurs : 100 positions, 4 orientations, 5 actions

    for _ in range(rep):
        spt, sot = env.reset()  # Reset the game, méthode reset definie ligne 85

        if (_+1) % 100 == 0 or _ > rep-2:
            print("{} sur {}".format(_ + 1, rep))  # avancement de l'entraînement
            print("{} = {}".format("Epsilon", epsilon))  # évolution d'epsilon

        if _ > rep-2:  # affiche la dernières parties
            print("reset")
            env.show()  # méthode show definie ligne 237

        epsilon = max(epsilon * (1 - 1 / (rep / 5)), 0.01)
        # Décroissant logarithmiquement, plus lent mais meilleur score
        # epsilon = max(epsilon - (1 / (rep/1.1)), 0.01)
        # Décroissant linéairement, plus rapide mais légèrement moins bon score
        # Rapport score/temps meilleurs avec une décroissance linéaire
        
        while not env.is_finished():  # méthode is_finished definie ligne 256
            at = take_action(spt, sot, Q, epsilon)  # fonction take_action definie ligne 260

            sptp1, sotp1, r = env.step(at)  # méthode step definie ligne 97
            # print("s", sptp1, sotp1)
            # print("r", r, "\n")

            # Update Q function, Q-Table ligne 279
            atp1 = take_action(sptp1, sotp1, Q, 0.0)  # fonction take_action definie ligne 260
            Q[spt][sot][at] = Q[spt][sot][at] + 0.1*(r + 0.9*Q[sptp1][sotp1][atp1] - Q[spt][sot][at])
            # 0.1 : learning rate  /  0.9 : gamma (moins d'importance aux actions lointaines)

            env.reward()  # méthode reward definie ligne 168
            if _ > rep - 101:
                score += r
                if _ > rep - 2:  # affiche la dernières parties
                    env.show()  # méthode show definie ligne 237
                    # epsilon = 0  # l'agent ne meurt jamais à la dernière partie s'il ne fait aucune action aléatoire

            spt = sptp1  # mise à jour de l'état
            sot = sotp1

    for s in range(11, 91):  # les états 1 à 11 et 90 à 100 ne sont jamais atteints
        print(s, Q[s])  # affiche la Q-Table

    print(score/100)  # score moyen sur les 100 dernières parties
