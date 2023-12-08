
responses = {}

responses['218022490'] = [
    (5, "You have a sufficient answer. I recommend numbering the figures so you"+\
        " don't have to use \"left\", \"right\", or \"other\" too much just to"+
        " navigate your viewers to the figures; this can reduce redundant verbiage"+
        " in your text. You should also label your plots and consider how viewers"+
        " may feel looking at your plots. For example, the x-axis text of your left"+
        " plot are really hard to read. Other than that, nice sufficient answer."),
    (5, "I liked your choice for a linear regression model and one-hot encoding does"+\
        " seem to best fit your feature types as well. I forgot but is the ratings"+
        " variable discrete? If so, classification would work. Though your choice"+
        " for linear regression + MSE is already great. Good job on this part!"),
    (6, "+1 BONUS point for accounting for multicollinearity. I liked the justifications"+\
        " you used for linear regression, though I think there many ways to make"+
        " this workable for a classification task. Nice consideration for multicollinearity"+
        " among the variables that may seem correlated with ratings as well."+ 
        " In future projects, I think incorporating a couple visuals showing some of"
        " these correlations would make your work much more compelling."),
    (7, "+2 BONUS points for in-depth and breadth-ful literature review. Although"+\
        " I liked your lit review a lot, it makes your paper comes across as severely"+
        " off-proportioned (section lengths-wise) and that there are many room for improvements that you"+
        " researched but didn't implement for your model. Great job reconnecting"+
        " back to your model on why certain things aren't applicable though."),
    (4, "-1 point for a lack of (1) alternative models and (2) parameter interpretations."+\
        " I personally wouldn't mention \"causality\" too much considering the scope"+
        " of your project because of how big of a topic jump that is. I recommend"+
        " interpreting your parameters, which can be as simple as spotting and identifying"+
        " the most influential (or least) params and their associated features. This will"+
        " not only help broaden your analysis but also help you understand your model"+
        " better. This is especially doable with models similar to yours, linear regression."),
]