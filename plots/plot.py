# bar plot to compare models
import matplotlib.pyplot as plt
import numpy as np

class Plot:    
    def plot(self,player1,player2,wins,loss,draws,dots=4):            
        
        #labels = [player1]
        #x = np.arange(len(labels))  # label locations
        x = 1
        width = 0.1  # width of each bar
        #games = wins+loss+draws
        # Plot
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width, wins, width, label='Player 1 Wins', color='green')
        rects2 = ax.bar(x, draws, width, label='Draws', color='gray')
        rects3 = ax.bar(x + width, loss, width, label='Player 2 wins', color='blue')

        # Labeling
        ax.set_ylabel('Number of Games')
        ax.set_title(f'{dots}x{dots} {player1} vs {player2}')
        ax.set_xlabel('')
        ax.legend()

        # Show numbers on top of bars (optional)
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{int(height)}',
                            xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        autolabel(rects3)

        plt.tight_layout()
        plt.savefig(f"plots/{dots}x{dots}{player2}vs{player1}.png")
        plt.show()