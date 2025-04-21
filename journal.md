# Weekly progress journal

## Instructions

In this journal you will document your progress of the project, making use of the weekly milestones.

Every week you should 

1. write down **on the day of the lecture** a short plan (bullet list is sufficient) of how you want to 
   reach the weekly milestones. Think about how to distribute work in the group, 
   what pieces of code functionality need to be implemented.
2. write about your progress **until Tuesday, 11:00** before the next lecture with respect to the milestones.
   Substantiate your progress with links to code, pictures or test results. Reflect on the
   relation to your original plan.

We will give feedback on your progress on Tuesday before the following lecture. Consult the 
[grading scheme](https://computationalphysics.quantumtinkerer.tudelft.nl/proj1-moldyn-grading/) 
for details how the journal enters your grade.

Note that the file format of the journal is *markdown*. This is a flexible and easy method of 
converting text to HTML. 
Documentation of the syntax of markdown can be found 
[here](https://docs.gitlab.com/ee/user/markdown.html#gfm-extends-standard-markdown). 
You will find how to include [links](https://docs.gitlab.com/ee/user/markdown.html#links) and 
[images](https://docs.gitlab.com/ee/user/markdown.html#images) particularly.

## Week 1
(due 22 April 2025, 11:00)

@npaarts

I started creating a basic structure of the polymer simulation.
After I finished doing that I continued with creating parts of the notes until I finally had enough.

During this time we implemented the end-to-end distance as a function of length

![](./journal/week1/end_to_end.png)

And also the gyration

![](./journal/week1/gyration.png)

The code is already using numpy with pre-allocated arrays where possible so it's quite performant.

@rjuyal

I reviewed the merge requests and added the errors(computed analytically) for the weighted averages graphs computed using Rosebluth method.

![](./journal/week1/end_to_end_error.png)

![](./journal/week1/gyration_error.png)

At some point we no longer see errors. This is probably because at that point, there is only 1 chain and since the analytical error formula has a N-1 term in the denominator, we get divide by zero. This is expected since we cannot really compute standard deviation of a single entry.

We are considering to look into the estimation(L$^{3/2}$) fitted by scipy and the errors associated with that as well.


## Week 2
(due 29 April 2025, 11:00)


## Week 3
(due 6 May 2025, 11:00)


## Reminder final deadline

The deadline for project 2 is **13 May 23:59**. By then, you must have uploaded the report to the repository, and the repository must contain the latest version of the code.
