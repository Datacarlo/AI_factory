"""Train this GAN using the common terminal-distribution pipeline."""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "learning/gan").is_dir())
sys.path.insert(0, str(ROOT))
from learning.gan.train import recipe_main

if __name__ == "__main__":
    recipe_main(Path(__file__).with_name("recipe.yaml"))
