import re
import time
import shutil
import string
import pathlib
import hashlib
import frontmatter


def get_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_md(file_path):
    content = ""
    metadata = {}
    with open(file_path) as f:
        post = frontmatter.load(f)
        content = post.content
        metadata = post.metadata
    return (content, metadata)


base_paths = [
    pathlib.Path("/home/yang/Dropbox/readings"),
]

forbidden_list = [
    "@Feng2020DeepGiniPrioritizingMassivea.md",
    "@Harel-Canada2020NeuronCoverageMeaningful.md",
    "@Muennighoff2023MTEBMassiveText.md",
    "@Ren2021SimpleFixMahalanobis.md",
    "@Ren2023OutofDistributionDetectionSelective.md",
    "@Rothermel1999TestCasePrioritization.md",
    "@Vidgen2021LearningWorstDynamically.md",
    "@Xie2023DataSelectionLanguage.md"
]

pattern = "\d{4}-\d{2}-\d{2}-(.*).md"
existing_name_dict = {
    re.search(pattern, filename.name)[1]: filename.stem
    for filename in pathlib.Path("_posts").glob("*.md")
}

for base_path in base_paths:
    for source_path in base_path.glob("*.md"):
        if source_path.name in forbidden_list:
            continue

        content, metadata = read_md(source_path)
        title = "-".join(re.findall("\w+", metadata["title"])).lower()

        # if making an update, then reusing the previous filename
        if title in existing_name_dict:
            filename = existing_name_dict[title]
        else:
            # if creating a new post, then using a new filename
            filename = "{}-{}".format(
                time.strftime('%Y-%m-%d'),
                title,
            )

        print("Moving {} to _posts directory".format(source_path.name))
        shutil.copyfile(source_path, f"_posts/{filename}.md")
