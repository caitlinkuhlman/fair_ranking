"Does mitigating ML impact disparity require treatment disparity"

They define 'treatment  disparity' in terms of using the protected attribute when designing the model, or making predictions. 
The idea is it may not be considered fair (or legal) to use the attribute to make decisions

'impact disparity' is an unfairness in the outcomes. they observe you naturally can use the protect attrivbute at the decision time
(introducing treatment disparity) in order to achieve fair outcomes. But if you try to avoid using the protected attribute at decision time, they show that you basically just end up
hiding the fact that the protected attribtue is in play. That is, unless the prot attr is completely independent of the rest of the data, then yo uwill use the protected attribute 
indirectly. They advocate for the transparent use of te protected attribute.

define use of prot. attr at training time but not prediction time "disparate learning processes" (dlp)

They make a key observation that Dlp's rely on the encoding of the protected attribute in the rest of the dataset to correct for the bias at decision time. They do this by advantaging 
objects that seem most like the protected group and disadvantaging the objects least like the protected group.

When the encoding is perfect, this is equivalent to just using the protected attribute at decision time. When the encoding is not perfect, then it leads to mistakes both in the protected and 
non-protected groups.

Perform an analysis on a real dataset made to be biased synthetically. (they flip 25% of the labels randomly) 




"Decoupled classifiers for group fair and efficient ML"

Techniques to best use protected attributes when making decisions. They qualify their work saying that this is not always appropriate or legal.

Basic idea is to train a separate set of classifiers for each group. REally they mean one classifier, with a decision threshold that can be varied 
to create different outcomes as in an ROC curve. Then they propose how to create a join loss funtion over these 2 sets which explicitly defines a tradeoff 
between fairness and accuracy. Then they have an exhaustive procedure for finding the best pair of classifiers which minimized the joint loss function.
They then propose to use transfer learning when training the 2 sets of classifiers to account for cases where one group is in a minority.


They are similar to us in 3 ways:

1. They correct for unfairness using different decision procedures for each group
2. They define loss functions corresponding to different fairness criteria. This is similarin spirit to us because they try to do it in a general way.
3. They evaluate their method empirically using synthetic data.

interesting notes that apply to us:
They make a caveat that their method may not be approporiate in all cases, and point out that in hiring, you can't legally use protected attributes when making hiring decisions.
So this is interesting as it applies to the Fa*ir paper. Their 'solution' really couldn't be applied in practice.

It is also true for us. This supports the idea that auditing is an important technique in itself, as you might not be able to correct using protected attributes directly.
Also, what about if the protected attribute is used during training, but not at the decision making time. Then are you ok leagaly? I think this is what the paper above suggests. 
In this case, can our method be adapted to do that? As in, maybe combining the groupwise calibrated models back together?
OR maybe it is interesting enough to note that if error is disproportionate for one group only, (like in our contrived example where one group in under-ranked) calibration will 
distribute the error among the groups ( this is like trading accuracy for one group for fairness. Interesting to demonstrate it but then want to do better. Can also denonstrate 
that calibrating by groups will improve the error for the one group without impacting the other. So it would be great to do that and then combine the results back together.)

Maybe we can Boost based on the error for the different groups
Or use an ensemble where the bagged samples are chosen based on group
What if you trained 2 regressors, and then always used the prediciton of the one that gave a better score? 

So if we have 
-c1 majority classifier (tends to undervalue members of the minority class) 
-c2 minority classifier (tends to treat minority fairly)

a \in majority
b \in minority

we predict 
c1(a), c2(a) - choose prediction which gives more preferred outcome
c1(b), c2(b) - chose prediciton which gives more preferred outcome

This shouldn't hurt accuracy for the majority class, and should treat minority class fairly, without explicitly using protected attribute. 

When would this fail? If the minority classifier overestimated members of the majority class

If we came up with a bunch of ways to try to correct for it without needing to use the protected attribute at decision time, and they compared their performance, that might be more interesting.

 
 




"Non-Discriminatory Machine Learning through Convex Fairness Criteria"
NOt sure what this journal/venue is. 

They do what we do defining various criteria, but they only look at eq odds and demographic parity.

They use the "p-rule" to validate whether soemthing is fair (this is analogous to the 80% rule).