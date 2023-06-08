from absl import app
from saxml.server.pax.lm.params import lm_cloud


class LmCloudSpmd2B(lm_cloud.LmCloudSpmd2B):
  ICI_MESH_SHAPE = [1, 1, 1]

  @property
  def test_mode(self) -> bool:
    # For checkpoint_path=None.
    return True


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  model_config = LmCloudSpmd2B()
  loaded = model_config.load(
      model_key="lm",
      checkpoint_path=None,
      primary_process_id=0,
      prng_key=0,
  )
  method = "lm.generate"
  method_obj = loaded.method(method)
  if method == "lm.gradient":
    inputs = [["hello world", "goodbye world"]]
  else:
    inputs = ["hello world"]
  print(method_obj.compute(inputs))


if __name__ == "__main__":
  app.run(main)
