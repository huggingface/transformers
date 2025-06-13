require 'dotenv'
Dotenv.load
require 'stripe'

Stripe.api_key = ENV['STRIPE_SECRET_KEY'] || 'sk_test_dummy'
Stripe.max_network_retries = 2
Stripe.api_version = '2020-08-27'