# Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead

Some variables do not need have gradient, just use var.detach() to remove it
