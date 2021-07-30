Since the data and simulations are in python we'll switch to python. Sadly this requires a lot of build-out of old ideas (though this is a different application anyway...)


Process:
1. "Sampling" data observations from each TC [Build out so we do multiple samplings] - look back at notes - what data can't we use... (was used in training of the model)
2. Selecting parameters
    a. Tuning of parameters:
        i. need to tune:
            1. the projection approach (let's use difusion mapping)
            2. sigma
            3. fixed radius value...
        ii. how to tune:
            1. mode clustering in this new space would work (1.,2.) *[from sklearn.cluster import MeanShift]*
            2. potentially the uniform distribution of conformal scores relative to some simulation distributions (3.)
    b. Other parameters that vary:
        i. # of simulations (probably 200, 1000 looks?)
3. Calibration set steps (and test points): capture random seed, create simulation (decide about storage potential?), calculate distances, calculate fixed radius value, projection & psuedo-density estimation for ranking,
    a. for cs score, for loop to check each point's minimum coverage distance (across all simulations) - can reduce space of checking relative to those already contained before. [https://github.com/benjaminleroy/EpiCompare/blob/93f203e031faa27828acb8c73dd2767a6b9a309e/R/conformal_prediction.R#L600 - try code in this area - shouldn't need to deal with Kmeans stuff - just straight matrix comparisons]
4. For test observations we'll just use the single average value for the cutoff (and also a single cutoff value) - for the average cutoff we need to do $\alpha/2$.
5. Visualizations: [https://plotly.com/python/3d-mesh/]
6. Diagnostics:
    a. for the test set:
        i. calculate containment at different alpha levels
        ii. maybe calculate the size (though this could be captured by average conformal score value of the test set?)
        iii. should think about this more...



Things to think about:
1. Parallelization / the amount of time to process a single observation...
    a. can we use PSC somewhat? [vera has 2 gpu nodes (twig and henon-gpu01) - requires "contact[ing] the relevant faculty member to be added to the requried allocations." - aka we might do a request from PSC itself - not vera...]
    b. can parallelization be done acceptably with just the Azure server?
2. Which observations should be the test set?
    a. I think this should be most recent observations
3. Presentation
    a. I think this also requires us to spell out what assumptions of exchangability we are using (especially with how we define test sets)







