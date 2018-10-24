# Systeme-de-reconnaissance-faciale
﻿Information sur l'application android :
 
		-Notre application portera comme nom Androidprojettensorflow et a comme icon celui d'android
		-On clique sur l'application, la premiere interface qui s'affiche contient 2 boutons (parcourir , detecter), une zone de text et une zone d'image
		-Le bouton parcourir nous emmene au galerie  ou on va choisir une image parmis les images de notre base de donnée. Si on choisit une image, l'image va s'fficher sur la zone image de notre application
		-Le bouton detecter nous permet de savoir la classe de cette image
		-On doit ajouter un module(jar/aar) à notre projet android et télécharger (tensorflow.aar) et l'ajouter à notre projet 

Information sur le script python :  

		-Le modèle utilisé est : AlexNet 
		-On a réussit à avoir une précision de 98% 
		-On a pu enregistrer les poids de notre modèle grace à la fonction export_model(), que vous allez trouvé dans le script tester.py et on a obtenu un fichier ".pb" qu'on va ajouter à l'application android 
