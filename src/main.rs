use anyhow::Result;
use ip2vec::dataset::Dataset;

fn main() -> Result<()> {
  let dataset = Dataset::import_dataset("../NF-UNSW-NB15-v3/data/NF-UNSW-NB15-v3.csv")?;

  Ok(())
}
