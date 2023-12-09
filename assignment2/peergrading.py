
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
    (6, "+1 bonus point for accounting for multicollinearity. I liked the justifications"+\
        " you used for linear regression, though I think there many ways to make"+
        " this workable for a classification task. Nice consideration for multicollinearity"+
        " among the variables that may seem correlated with ratings as well."+ 
        " In future projects, I think incorporating a couple visuals showing some of"
        " these correlations would make your work much more compelling."),
    (7, "+2 bonus points for in-depth and breadth-ful literature review. Although"+\
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

responses['217709846'] = [
    (6, "+1 bonus point for highly-detailed introduction to Steam. Nice introduction to your project,"+
        " I am sure some readers would appreciate"+\
        " your description of Steam's history. Great clarifications on how"+
        " you made the positive and negative data samples. I find your ratio of"+
        " train/valid/test split pretty interesting, I guess that could work with"+
        " such a large number of data points. The only recommendation I have is"+
        " too include more informative and \"fun\" visualizations instead of a python"+
        " list of lists."),
    (4, "-1 point for insufficient model comparisons, you sort of did, but lack strengths"+\
        " and weaknesses for each model relative to one another. Overall, solid analysis"+
        " and assessment of your model's objective value per iteration, though I think"+\
        " line graphs would look much more appealing than boxes of numbers/text. Great"+
        " job overall though."),
    (5, "Since you sort of combined section 2 and 3, my feedback here will be the same"+\
        " as above. One final recommendation I have in this section is to have all of your"+
        " accuracies for each model in one table for easier comparison by the readers."),
    (7, "+2 bonus points for a detailed literature review. I really enjoyed reading"+\
        " your literature review; however, for each paper, I recommend briefly describing the gist"+
        " behind each model used (or only describe the best model), rather than listing"+\
        " them side-by-side (sometimes this looks like a letter dump)."),
    (5, "Your answer is sufficient. Please also ignore my recommendation regarding"+
        " an accuracy table in section 3 as you did that here. Good work overall, I enjoyed reading your paper.")
]

responses['218001062'] = [
    (5, "Good statistical and visual analysis overall. I recommend emphasizing"+\
        " the \"meaning\" of your statistical observations, rather than listing"+
        " the numbers or positions of the plot components because the readers can see"+
        " these for themselves, or a combination of both. For example, I think"+\
        " lines like \"Figure 1.52 shows that the higher the payment, the more polarized"+
        " the stars of the item tend to be\" is beautiful, it directly shows the meaning"+
        " of what is seen."),
    (5, "Nice explanations for why you used MSE, though was it necessary to include"+\
        " the MAE equation since you are not going to use it? Nice consideration of"+
        " generalizability (or applicability) as well. Good logic towards the end"+
        " too where you considered the variables' availability and chronological"+
        " order in actual practice."),
    (6, "+1 bonus point for comprehensiveness. Great and concise model descriptions"+\
        " for this section! I really liked Figure 3.31 that provided a brief"+
        " description for each model provided by Surprise, this allwos people who"+
        " are new to the Surprise library understand these models. I liked the"+
        " diversity in model types as well, great breadth and depth of analysis here."),
    (5, "Good literature review, it reminded me of one of the last lectures."+\
        " Nice coverage of the papers and the gist of their ideas, I find your"+
        " summaries easy to read and they also provide a comparative picture between"+
        " the models well."),
    (5, "Nice and long closing; it allows a reader to get a general grasp of your"+\
        " paper without reading any other section. I liked your connection to"+
        " real-world applications towards the end as well to better connect customers"+
        " and items.")
]

responses['218052927'] = [
    (5, "Nice and sufficient introduction; I would appreciate a little more exploratory"+\
        " analysis and statistical properties descriptions though. Good abstract as well."+
        " I am curious that if the amount of zero helpfulness was to the point of"+
        " inability for removal, how bad was your data imbalance problem?"),
    (5, "FYI, I stopped before section 3 for the second section of the assignment"+\
        " write up. You have good model descriptions and evaluations. I recommend"+
        " having one uniform language and tone throughout your paper will help by"+
        " not giving readers a language-shock. I loved how you included"+
        " \"if you're following along\". Also, 8 x 10^-5 is a really small number"+
        " to talk about correlations in the context as you did."),
    (5, "Nice visualizations and tables showing your models' performance. I feel bad"+
    " for your experiments section as it deserves a similar length as other sections :("),
    (5, "Solid literature review, you presented a wide variety of modeling methods"+
    " and features choices that are available for the task."),
    (5, "Good conclusion. I liked how you connected your paper's findings to real-world"+
    " applications of your use case. Nice job.")
]

f = open('responses.txt', 'w')
f.write(str(responses) + '\n')
f.close()