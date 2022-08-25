# PyWhy static website (Jekyll)

This repo holds the source code and content to pywhy.org. 

## Editing Content

For significant changes, follow [GitHub Flow](https://guides.github.com/introduction/flow/): create a new branch and submit a pull request to get review feedback from other team members. Once the PR is merged to main, the site will automatically rebuild and deploy.

The site is compiled using [Jekyll](https://jekyllrb.com/), so all of the content is written in [Markdown](https://daringfireball.net/projects/markdown/). In particular, Jekyll uses a superset of Markdown called Kramdown. [Syntax documentation here](https://kramdown.gettalong.org/syntax.html).

### File struture

The content is structured to avoid needing HTML editing for the site content. There are a few places content resides in order to be as obvious as possible while also adhering to Jekyll requirements.

- Main pages (root): any *.md files in the root of the project directory are top-level pages (except this README!). Most pages are Markdown except for the main home page (index.html) because it requires advanced HTML layout that Markdown does not support. *.md files will be converted to HTML files automatically for the deployment.
- Content chunks [/content](./content): Fragments of markdown content to be included in other markdown or html pages. Content markdown files do not contain YAML Front Matter as they are not standalone pages. 
- Collections: represent types of content pages much like blog articles or news stories. Collections in this site include case studies articles (\_case_studies), learn content (\_learn), and news articles (\_news).

### YAML Front Matter

Every page has YAML front matter that gives Jekyll some instructions on how to process it:
- title: displays a large-format title on the page.
- layout: indicates what template should be used to render the page. Most pages will use the "page" template.
- description: display a short description within cards. 
- summary: a longer summary of the resource displayed in aggregate view (news page, case studies page, etc).
- image: optional large-format image to include in the layout beackground behind the title.
- image-alt: alt attribute for the image and must be present for the image to be shown.
- slug: url friendly name for the page.
- link: An optional link to an external resource. If present, the link attribute indicates that the page/resource
  is external and generated cards and summaries will link out to the external resource instead of a local page.

## Editing the Site Locally

For deeper edits or site changes, you'll want to clone the repo and run Jekyll locally:

- Install [Docker](https://www.docker.com/get-started) for your platform and make sure it is running
- Install [Docker Compose](https://docs.docker.com/compose/install/) for your platform
- From a terminal, `cd` into the cloned repo directory and run `docker-compose up`

This will download the default Jekyll Docker container and spin up the Jekyll instance. Once it's ready (it might take a few minutes the first time), you can view the site at http://localhost:4000. The `docker-compose.yml` includes live reload config, so as you edit the site it will automatically refresh in your browser.

### Removing pages

If you need to remove a page, you'll want to redirect to preserve the links and prevent 404s for users that may have bookmarks.

Use the Jekyll [redirect plugin](https://github.com/jekyll/jekyll-redirect-from). If you choose a new page that the user should be directed to, you can add front matter to the new page that configures redirects, then delete the old page. It looks something like this:

```
redirect_from:
  - /data-collection.html
```
