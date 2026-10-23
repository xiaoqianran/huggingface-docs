# Collections

Use Collections to group repositories from the Hub (Models, Datasets, Spaces and Papers) on a dedicated page.

![Collection page](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-intro.webp)

Collections have many use cases:

- Highlight specific repositories on your personal or organizational profile.
- Separate key repositories from others for your profile visitors.
- Showcase and share a complete project with its paper(s), dataset(s), model(s) and Space(s).
- Bookmark things you find on the Hub in categories.
- Have a dedicated page of curated things to share with others.
- Gate a group of models/datasets (Team & Enterprise)

This is just a list of possible uses, but remember that collections are just a way of grouping things, so use them in the way that best fits your use case.

## Creating a new collection

There are several ways to create a collection:

- For personal collections: Use the **+ New** button on your logged-in homepage (1).
- For organization collections: Use the **+ New** button available on organizations page (2).

![New collection](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-new.webp)

It's also possible to create a collection on the fly when adding the first item from a repository page, select **+ Create new collection** from the dropdown menu.
You'll need to enter a title and short description for your collection to be created.

## Adding items to a collection

There are 2 ways to add items to a collection:

- From any repository page: Use the context menu available on any repository page then select **Add to collection** to add it to a collection (1).
- From the collection page: If you know the name of the repository you want to add, use the **+ add to collection** option in the right-hand menu (2).

![Add items to collections](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-add.webp)

It's possible to add external repositories to your collections, not just your own.

## Collaborating on collections

Organization collections are a great way to build collections together. Members with Read-only access can view collections, but only members with Write (or higher) organization permissions can create collections or add, edit, and remove items.
Use the **history feature** to keep track of who has edited the collection.

![Collection history](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-history.webp)

## Collection options

### Collection visibility

![Collections on profiles](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-profile.webp)

**Public** collections appear at the top of your profile or organization page and can be viewed by anyone. The first 3 items in each collection are visible directly in the collection preview (1). To see more, the user must click to go to the collection page.

Set your collection to **private** if you don't want it to be accessible via its URL (it will not be displayed on your profile/organization page). For organizations, private collections are only available to members of the organization.

### Gating Group Collections (Team & Enterprise)

You can use a collection to [gate](https://huggingface.co/docs/hub/en/models-gated) all the models/datasets belonging to it, allowing you to grant (or reject) access to all of them at once.

This feature is reserved for [Team & Enterprise](https://huggingface.co/docs/hub/en/enterprise) subscribers: more information about Gating Group Collections can be found in [our dedicated doc](https://huggingface.co/docs/hub/en/enterprise-gating-group-collections).

### Assigning a collection to a Resource Group (Team & Enterprise)

Organization collections can be assigned to a [Resource Group](./security-resource-groups) to control which members can access them, the same way repositories are. You can pick a Resource Group when creating an organization collection, and members with the appropriate permissions can move a collection to a different Resource Group later from the collection's menu. Private collections that belong to a Resource Group are only visible to members of that group.

This feature is reserved for [Team & Enterprise](https://huggingface.co/docs/hub/en/enterprise) subscribers.

> [!NOTE]
> Assigning a collection to a Resource Group controls access to *that specific collection*. To restrict who can create or edit organization Collections at all, see [Granular feature access](./security-resource-groups#granular-feature-access) (Enterprise plans and above).

### Ordering your collections and their items

You can use the drag and drop handles in the collections list (on the left side of your collections page) to change the order of your collections (1). The first two collections will be directly visible on your profile/organization pages.

You can also sort repositories within a collection by dragging the handles next to each item (2).

![Collections sort](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-sort.webp)

### Deleting items from a collection

To delete an item from a collection, click the trash icon in the menu that shows up on the right when you hover over an item (1).
To delete the whole collection, click delete on the right-hand menu (2) - you'll need to confirm this action.

![Collection delete](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-delete.webp)

### Adding notes to collection's items

It's possible to add a note to any item in a collection to give it more context (for others, or as a reminder to yourself). You can add notes by clicking the pencil icon when you hover over an item with your mouse. Notes are plain text and don't support markdown, to keep things clean and simple. URLs in notes are converted into clickable links.

![Collection note](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collection-note.webp)

### Adding images to a collection item

Similarly, you can attach images to a collection item. This is useful for showcasing the output of a model, the content of a dataset, attaching an infographic for context, etc.

To start adding images to your collection, you can click on the image icon in the contextual menu of an item. The menu shows up when you hover over an item with your mouse.

![Collection image icon](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collections-image-button.webp)

Then, add images by dragging and dropping images from your computer. You can also click on the gray zone to select image files from your computer's file system.

![Collection image drop zone with images](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collections-image-gallery.webp)

You can re-order images by drag-and-dropping them. Clicking on an image will open it in full-screen mode.

![Collection image viewer](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/collections/collections-image-viewer.webp)

## Your feedback on collections

We're working on improving collections, so if you have any bugs, questions, or new features you'd like to see added, please post a message in the [dedicated discussion](https://huggingface.co/spaces/huggingface/HuggingDiscussions/discussions/12).

### TF-Keras (legacy)
https://huggingface.co/docs/hub/tf-keras.md

## TF-Keras (legacy)

`tf-keras` is the name given to Keras 2.x version. It is now hosted as a separate GitHub repo [here](https://github.com/keras-team/tf-keras). Though it's a legacy framework, there are still [4.5k+ models](https://huggingface.co/models?library=tf-keras&sort=trending) hosted on the Hub. These models can be loaded using the `huggingface_hub` library. You **must** have either `tf-keras` or `keras<3.x` installed on your machine.

If you are interested in Keras 3.x support, check out [this guide](./keras).

Once installed, you just need to use the `from_pretrained_keras` method to load a model from the Hub. Read more about `from_pretrained_keras` [here](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/mixins#huggingface_hub.from_pretrained_keras).

```py
from huggingface_hub import from_pretrained_keras

model = from_pretrained_keras("keras-io/mobile-vit-xxs")
prediction = model.predict(image)
prediction = tf.squeeze(tf.round(prediction))
print(f'The image is a {classes[(np.argmax(prediction))]}!')

<CopyLLMTxtMenu containerStyle="float: right; margin-left: 10px; display: inline-flex; position: relative; z-index: 10;"></CopyLLMTxtMenu>

# The image is a sunflower!
```

You can also host your `tf-keras` model on the Hub. However, keep in mind that `tf-keras` is a legacy framework. To reach a maximum number of users, we recommend to create your model using Keras 3.x and share it natively as described above. For more details about uploading `tf-keras` models, check out [`push_to_hub_keras` documentation](https://huggingface.co/docs/huggingface_hub/main/en/package_reference/mixins#huggingface_hub.push_to_hub_keras).

```py
from huggingface_hub import push_to_hub_keras

push_to_hub_keras(model,
    "your-username/your-model-name",
    "your-tensorboard-log-directory",
    tags = ["object-detection", "some_other_tag"],
    **model_save_kwargs,
)
```

## Additional resources

- [GitHub repo](https://github.com/keras-team/tf-keras)
* Blog post [Putting Keras on 🤗 Hub for Collaborative Training and Reproducibility](https://merveenoyan.medium.com/putting-keras-on-hub-for-collaborative-training-and-reproducibility-9018301de877) (April 2022)

### Quickstart
https://huggingface.co/docs/hub/jobs-quickstart.md
