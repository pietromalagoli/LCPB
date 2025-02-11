9/02
Al momento quello il file che funziona correttamente (MSE=0.07 con lat_dim=4, tutte le cartelle e tutte le features) è
eugi_all_features_logenergy.py.

Quello che faccio ora è pulire il file e probabilmente io consegnerei quello, senza cercare di dividerlo in altri file ausiliari.

Ora il file è pulito e funziona bene. faccio una run per i grafici più tardi.

10/02

Sto scrivendo il report e facendo un po' di run.
C'è da controllare come viene calcolato l'MSE.
Per l'MSE ho cambiato e ora prende la media dell'MSE su tutti i profili, quando invece prima prendeva solo sul primo. 
Ho fatto questo cambiamento perché non c'è un motivo valido per cui dovremmo guardare solo il primo profilo quando valutiamo le performance
del modello. Dobbiamo vedere come il modello performa su tutti i profili. 

11/02 
finisco di fare il codice per i plot. quasi fatto
si può fare anche un plot dell'mse per varie latent dimension per one feature e poi per 4