var AdaBoost = require('./AdaBoost');


(function () {
	'use strict';

	var trainingData = [
		["Coughing", "Male", "Adult", "Discharged"],
		["Coughing", "Female", "Teen", "Discharged"],
		["Headache", "Male", "Child", "Discharged"],
		["Headache", "Male", "Teen", "Discharged"],
		["Hiccups", "Female", "Adult", "Discharged"],
		["Sneezing", "Male", "Teen", "Discharged"],
		["Sneezing", "Female", "Child", "Admitted"],
		["Sneezing", "Male", "Child", "Admitted"],
		["Hiccups", "Female", "Teen", "Admitted"],
		["Coughing", "Female", "Adult", "Admitted"]
	];

	var adaBoost = new AdaBoost(trainingData, "Discharged");

	console.log(adaBoost.trainedLearners);
	console.log("");

	var vote = adaBoost.predict(["Coughing", "Male", "Child"],true);
	console.log("The final vote is " + vote, (vote == -0.7156340528869896) );
	console.log("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));
	console.log("");

	vote = adaBoost.predict(["Headache", "Female", "Child"]);
	console.log("The final vote is " + vote, (vote == 1.826510427853278)); // 
	console.log("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));
	console.log("");

	vote = adaBoost.predict(["Hiccups", "Female", "Adult"]);
	console.log("The final vote is " + vote, (vote == 0.0407126627958313));
	console.log("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));
	console.log("");

	vote = adaBoost.predict(["Headache", "", "Child"]);
	console.log("The final vote is " + vote, (vote == 2.4140941225257038));
	console.log("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));

	

	/*
	The final vote is -0.7156340528869896
	The final vote is 1.826510427853278
	The final vote is 0.0407126627958313
	The final vote is 2.4140941225257038

	Coughing predicted 1 and had an weight of 0.626918466904179
	Male predicted 1 and had an weight of 3.151108136665014
	Child predicted -1 and had an weight of 4.493660656456183
	The final vote is -0.7156340528869896
	The person will most likely be Admitted

	Headache predicted 1 and had an weight of 6.907754778981887
	Female predicted -1 and had an weight of 0.5875836946724257
	Child predicted -1 and had an weight of 4.493660656456183
	The final vote is 1.826510427853278
	The person will most likely be Discharged

	Female predicted -1 and had an weight of 0.5875836946724257
	Adult predicted 1 and had an weight of 0.628296357468257
	The final vote is 0.0407126627958313
	The person will most likely be Admitted

	Headache predicted 1 and had an weight of 6.907754778981887
	Child predicted -1 and had an weight of 4.493660656456183
	The final vote is 2.4140941225257038
	The person will most likely be Discharged
	 */
})();