# Visual Search at Batiplus

## Goal

Visual search is becoming more and more useful in today's world.
People qant to take pictures of furniture they like, and be reocmmended similar ones.

It's used in a lot of apps : Zalando, Farfetch, Pinterest, Snapchat, etc

## How

Compute features of all images inside Odoo

Image can be in multiple products, so use MD5/SHA sum to make it unique (as an ID).
Then for each sum, link the associated products.

Detect object in images, and compute features of only those objects (within the bounding box of the object in the image).

When a user upload an image, compute it's features too.

Then use distance comparison (Approximate Nearest Neighbors) to find similar images/objects.