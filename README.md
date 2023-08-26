# Automated Blog Publishing on WordPress

## Overview

I copy this project directly from [here](https://github.com/zhaoolee/WordPressXMLRPCTools) with no changes. I only rewrite the `README.md` for clarity. This project uses GitHub Actions and XMLRPC (supported by WordPress by default at `https://example.com/xmlrpc.php`) to automatically update the posts using **one command**.

## Blog Structure

A blog post is a `.md` file, and it should stay in the `_posts` directory. Each post should follow the template below:

- Make sure there are absolutely no ":" in the `<title>` otherwise the automated posting WILL fail.
- Tags act like keywords in research papers.
- Categories are the selectors for a quick location of a specific post.

```markdown
---
title: <title>
tags: 
- <tag 0>
- <tag 1>
- <tag 2>
categories:
- <category 1>
- <category 2>
---

Four score and seven years ago our fathers brought forth on this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.

Now we are engaged in a great civil war, testing whether that nation, or any nation so conceived and so dedicated, can long endure. We are met on a great battle-field of that war. We have come to dedicate a portion of that field, as a final resting place for those who here gave their lives that that nation might live. It is altogether fitting and proper that we should do this.

But, in a larger sense, we can not dedicate—we can not consecrate—we can not hallow—this ground. The brave men, living and dead, who struggled here, have consecrated it, far above our poor power to add or detract. The world will little note, nor long remember what we say here, but it can never forget what they did here. It is for us the living, rather, to be dedicated here to the unfinished work which they who fought here have thus far so nobly advanced. It is rather for us to be here dedicated to the great task remaining before us—that from these honored dead we take increased devotion to that cause for which they gave the last full measure of devotion—that we here highly resolve that these dead shall not have died in vain—that this nation, under God, shall have a new birth of freedom—and that government of the people, by the people, for the people, shall not perish from the earth.
```

The link format could be customized in WordPress control panel: "Settings" $\rightarrow$ "Permalinks". 

## GitHub Actions Setup

Setting `USERNAME`, `PASSWORD`, and `XMLRPC_PHP` at "Settings" $\rightarrow$ "Secrets" $\rightarrow$ "Repository Secrets" in GitHub. This guarantees that only GitHub can access this information.

- `USERNAME`, `PASSWORD`: The username and password used to log into `https://example.com/wp-admin`.
- `XMLRPC_PHP`: `https://example.com/xmlrpc.php`.

## Usage

- Step 1: Add a new article or update an existing one in `_posts` following the blog structure explained above. The following is a command that makes sure the contents of two folders are exactly the same ([answer](https://unix.stackexchange.com/a/203854/307215)):

  ```bash
  rsync -avu --delete <src> <tgt>
  ```

- Step 2: Run the following command, and you will see that (1) the `README.md` is updated with prepended blog links sorted in reverse chronological order, and (2) the GitHub Actions will show a record. This may take **1 to 2 minutes** to take effect.

  ```bash
  git pull && git add _posts && git commit -m "update" && git push
  ```


  
