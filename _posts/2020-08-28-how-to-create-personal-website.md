---
layout: post
title:  "How to create a personal website with GitHub Pages"
date:   2020-08-28
image:  images/04.jpg
tags:   [study]
---

Hello, welcome to my blog! This post will introduce how to create a personal website with **[GitHub Pages][GitHub Pages]** and site generator **[Jekyll][Jekyll]**.

GitHub: https://github.com/eveyimi/eveyimi.github.io

# Create a repository
The fist thing you need to do is to go to GitHub and create a new repository named *username*.github.io, where *username* is your username on GitHub. If the first part of the repository doesn’t exactly match your username, it won’t work, so make sure to get it right.

![Create a new repository]({{site.baseurl}}/images/HW1/repo.png)
*Create a new repository*

And then you can clone it to your local machine.

{% highlight shell %}
~ $ git clone https://github.com/username/username.github.io.git
{% endhighlight %}

<br>

# Combine a Jekyll theme
Jekyll is a simple, blog-aware, static site generator for personal, project, or organization sites. You can also choose Pelican or Hugo. Here I will use Jekyll. You can go to the official website of Jekyll for more themes.

## Set up enviroment

To run Jekyll, you need to set up enviroment first. Follow the instruction below to install required package. Once you finish it, you can browse *http://localhost:4000* and check your personal website.

{% highlight shell %}
~ $ gem install bundler jekyll
{% endhighlight %}

If you want to create a simple Jekyll theme you can do that as followed. But here I will use a published theme instead. 

{% highlight shell %}
~ $ gem install bundler jekyll
~ $ jekyll new my-awesome-site
~ $ cd my-awesome-site
~/my-awesome-site $
~ $ bundle exec jekyll serve
# => Now browse to http://localhost:4000
{% endhighlight %}

## Download a Jekyll theme

You can select a Jekyll theme from its official website. I selected my theme from *https://jekyll-themes.com* and the theme name is **[Reked][Reked]**. You can both download or `git clone`. Here I chose to download it and initialize the local directory as a Git repository. Please follow the below instructions and remember to `unzip` the `theme.zip` first and go inside the folder using `cd`.

{% highlight shell %}
~ $ git init
~ $ git add .
~ $ git commit -m "First commit"
~ $ git remote add origin https://github.com/username/username.github.io.git
~ $ git remote -v
~ $ git push -u origin master
{% endhighlight %}

## Edit configuration file

We also need to change the configuration to your own information by editting `_config.yml`. There are several settings you can change according to your preference.

{% highlight yml %}
title: Blog # The title of the blog.
logo: "images/logo.svg" # You can add own logo image.
description: Personal Website for BIOSTAT 823 in 2020 Fall. # Description.
baseurl: "" # The subpath of your site, e.g. /blog
url: "https://eveyimi.github.io" # The base hostname & protocol for your site.

# Author Settings
author:
  name: 
  bio: Thank you for visiting my blog.

# Contact links
twitter: https://twitter.com/ # Add your Twitter handle
facebook: https://facebook.com/ # Add your Facebook handle
dribbble: https://dribbble.com/ # Add your Dribbble handle
instagram: https://instagram.com/ # Add your Instagram handle
pinterest: https://pinterest.com/ # Add your Pinterest handle
email: forexample@website.com # Add your Email address

# Hero Section
hero: true # To enable the section hero, use the value true. To turn off use the value false.
hero-title: Welcome # Add your hero title
hero-subtitle: This is Yi's Personal website # Add your hero subtitle
hero-image: images/14.jpg # Add background image in section hero

# Footer
footer-image: images/14.jpg # Add background image in footer
....

{% endhighlight %}


## Run on your local machine

Once you finish the instructions above, you are ready to run your personal website at *http://localhost:4000*, which can be automatically rendered each time you edit and save.

{% highlight shell %}
~ $ bundle exec jekyll serve
# => Now browse to http://localhost:4000
{% endhighlight %}

<br>

# Add new posts

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.md` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works. You can also refer to `Style Guide` page for more inforamtion about styling and formats.

<br>

# Publish your website

Once you push your code to the repository, you're done! Fire up a browser and go to https://username.github.io.

<br>


[GitHub Pages]: https://pages.github.com/
[Jekyll]: https://jekyllrb.com/
[Reked]: https://jekyll-themes.com/reked/