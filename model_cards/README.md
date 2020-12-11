## 🔥 Model cards now live inside each huggingface.co model repo 🔥


For consistency, ease of use and scalability, `README.md` model cards now live directly inside each model repo on the HuggingFace model hub.

### How to update a model card

You can directly update a model card inside any model repo you have **write access** to, i.e.:
- a model under your username namespace
- a model under any organization you are a part of.

You can either:
- update it, commit and push using your usual git workflow (command line, GUI, etc.)
- or edit it directly from the website's UI.

**What if you want to create or update a model card for a model you don't have write access to?**

In that case, given that we don't have a Pull request system yet on huggingface.co (🤯),
you can open an issue here, post the card's content, and tag the model author(s) and/or the Hugging Face team.

We might implement a more seamless process at some point, so your early feedback is precious!
Please let us know of any suggestion.

### What happened to the model cards here?

We migrated every model card from the repo to its corresponding huggingface.co model repo. Individual commits were preserved, and they link back to the original commit on GitHub.
