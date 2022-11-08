# OCP5-Customer-segmentation
Segmentation des clients d'un site de e-commerce

Ce projet met en œuvre la problématique de segmentation des clients à partir d'un ensemble de 9 tables de données en utilisant des algorithmes de clustering (apprentissage non supervisé).
La segmentation est centrée sur l'approche marketing RFM (Récence, Fréquence et Montant des commandes).

Les tables de données sont assemblées pour constituer le jeu de données global. Les valeurs manquantes sont traitées et le jeu de données est analysé de manière univariée et multivariée afin de retenir et produire les features les plus utiles à la segmentation.
Les features (numériques et catégorielles) sont transformées afin de pouvoir être exploitées par des algorithmes de clustering (basés sur la distance).

L'analyse en composantes principales et la réduction de dimension (PCA et t-SNE) permettent d'anticiper les segmentations les plus pertinentes et de les visualiser graphiquement.
Plusieurs algorithmes de clustering sont testés: k-Means, DBScan, AgglomerativeClustering / clustering hiérarchique, et modèle de mélange gaussien (GMM).
Au delà des features RFM de base, d'autres features sont ajoutées pour en examiner les effets sur la segmentation.

La segmentation est analysée sous l'angle métier pour en valider la pertinence et l'utilité pour l'action marketing de l'entreprise.
La durée de vie du modèle de segmentation est estimée afin d'examiner sa cohérence avec l'échelle de temps des actions marketing et anticiper sa maintenance.