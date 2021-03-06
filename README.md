
[![Build Status](https://travis-ci.org/TravCav/AdaBoost.svg?branch=master
)](https://travis-ci.org/TravCav/AdaBoost)
[![npm version](https://badge.fury.io/js/adaboost.svg)](https://badge.fury.io/js/adaboost)

# AdaBoost
[![Join the chat at https://gitter.im/TravCav/AdaBoost](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/TravCav/AdaBoost?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)



Adaptive Boost Algorithm

# Example
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
		["Coughing", "Female", "Adult", "Admitted"],
		["Coughing", "Male", "Child", "Admitted"]
	];

	var adaBoost = new AdaBoost(trainingData, "Discharged");

	var vote = adaBoost.predict(["Coughing", "Male", "Child"], true);
	console.log("The final vote is " + vote);
	console.log("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));
	console.log("");
	// The final vote is -1.7181763612259795
	// The person will most likely be Admitted

	vote = adaBoost.predict(["Headache", "Female", "Child"]);
	console.log("The final vote is " + vote); 
	console.log("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));
	console.log("");
	// The final vote is 1.60653649762956
	// The person will most likely be Discharged

	vote = adaBoost.predict(["Hiccups", "Female", "Adult"]);
	console.log("The final vote is " + vote);
	console.log("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));
	console.log("");
	// The final vote is -0.0843346278612036 true
	// The person will most likely be Admitted

	vote = adaBoost.predict(["Headache", "", "Child"]);
	console.log("The final vote is " + vote);
	console.log("The person will most likely be " + (vote > 1 ? "Discharged" : "Admitted"));
	console.log("");
	// The final vote is 2.3552095483256945 true
	// The person will most likely be Discharged