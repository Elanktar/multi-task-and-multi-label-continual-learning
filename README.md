# Dépôt github et matériel supplémentaire du papier "Apprentissage multi-labels et multi-tâches en continu pour données tabulaires: proposition d'un protocole de création de tâches et d'évaluation de classifieurs".
___

## Données :
- Clusters : contient les tâches créées à partir des différents jeux de données
- Eval_set : contient les ensembles d'évaluation des différentes tâches
- Experiences : contient les expériences d'apprentissage
- Labels : indique les signatures de label de chaque tâche
- Length : indique le nombre d'instance pour chaque expérience d'apprentissage
- Orders : contient l'ordre des expériences d'apprentissage pour chaque flux
- datasets : contient les jeux de données dans leur état initial

___

## Résultats :
- bench_metrics : contient le code des métriques utilisées
- consumption : contient les fichiers de sortie de code carbon et les tableaux générés concernant la frugalité
- graphs : contient tous les graphiques générés
- results : contient tous les résultats ainsi que des tables de résultats

___

## Modèles :
- Config : contient les configurations d'hyperparamètres retenues par la HPO pour chaque modèle et chaque flux
- implemented_models : contient les modèles implémentés

___

## Scripts :
Pour chaque script, taper dans un terminal linux "bash nom_fichier.sh"
Les paramètres du script python associé sont à changer dans le fichier .sh
Le fichier parameters.py doit être à jour avec les noms des modèles à tester, ainsi que ceux des jeux de données, le nombre d'attributs et le nombre de labels.

- 1_task_generator : créer les tâches à partir d'un jeu de données
- 2_order_generator : génère un ordre pour les tâches
- 3_benchmark : mène l'HPO, ainsi que l'évaluation des modèles sur les flux (indiqués dans parameters.py)
- 4_graph_generator : génère les figures pour les métriques en ligne
- 5_CL_eval_process : génère les figures pour les métriques d'évaluation continue
- 6_table_generator : génère les tableaux de résultats
