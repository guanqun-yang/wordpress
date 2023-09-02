import time
import shutil
import pathlib
import frontmatter


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


    
for base_path in base_paths:
    for source_path in base_path.glob("*.md"):
        if source_path.name in forbidden_list:
            continue

        print(source_path)
        
        _, metadata = read_md(source_path)
        filename = "{}-{}".format(
            time.strftime('%Y-%m-%d'),
            metadata["title"].replace(" | ", "-").replace(" - ", "-").replace(" ", "-")
        ).lower()

        print("Moving {} to _posts directory".format(source_path.name))
        shutil.copyfile(source_path, f"_posts/{filename}.md")