# What does this PR do?


According to <do_eval> description, when <evaluation_strategy> is different from <no> it will be set <True>.
But the <do_eval>'s defualt setting is None. So, this code can't be executed unless the the user set <do_eval = False>.
'''
if self.do_eval is False and self.evaluation_strategy != IntervalStrategy.NO:
      self.do_eval = True  
'''
I think it will be better to change <do_eval>'s defulat value <None> into <False>

- How I FOUND IT.
I was trying to use <training_args.do_eval> in my script.
BUT it didn't worked even the <evaluation_strategy> was set to <steps>.


