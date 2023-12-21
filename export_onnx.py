import onnxruntime
import torch
import torch.nn.functional as F

from efficient_sam.build_efficient_sam import build_efficient_sam_vitt


class OnnxEfficientSam(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def decoder_max_num_input_points(self):
        return self.model.decoder_max_num_input_points

    @property
    def get_image_embeddings(self):
        return self.model.get_image_embeddings

    @property
    def prompt_encoder(self):
        return self.model.prompt_encoder

    @property
    def mask_decoder(self):
        return self.model.mask_decoder

    def forward(
        self,
        batched_images: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
    ):
        batch_size, _, input_h, input_w = batched_images.shape
        image_embeddings = self.get_image_embeddings(batched_images)
        return self.predict_masks(
            image_embeddings,
            batched_points,
            batched_point_labels,
            multimask_output=True,
            input_h=input_h,
            input_w=input_w,
            output_h=input_h,
            output_w=input_w,
        )

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        batched_points: torch.Tensor,
        batched_point_labels: torch.Tensor,
        multimask_output: bool,
        input_h: int,
        input_w: int,
        output_h: int = -1,
        output_w: int = -1,
    ):
        batch_size, max_num_queries, num_pts, _ = batched_points.shape
        num_pts = batched_points.shape[2]
        rescaled_batched_points = batched_points

        if num_pts > self.decoder_max_num_input_points:
            rescaled_batched_points = rescaled_batched_points[
                :, :, : self.decoder_max_num_input_points, :
            ]
            batched_point_labels = batched_point_labels[
                :, :, : self.decoder_max_num_input_points
            ]
        elif num_pts < self.decoder_max_num_input_points:
            rescaled_batched_points = F.pad(
                rescaled_batched_points,
                (0, 0, 0, self.decoder_max_num_input_points - num_pts),
                value=-1.0,
            )
            batched_point_labels = F.pad(
                batched_point_labels,
                (0, self.decoder_max_num_input_points - num_pts),
                value=-1.0,
            )

        sparse_embeddings = self.prompt_encoder(
            rescaled_batched_points.reshape(
                batch_size * max_num_queries, self.decoder_max_num_input_points, 2
            ),
            batched_point_labels.reshape(
                batch_size * max_num_queries, self.decoder_max_num_input_points
            ),
        )
        sparse_embeddings = sparse_embeddings.view(
            batch_size,
            max_num_queries,
            sparse_embeddings.shape[1],
            sparse_embeddings.shape[2],
        )
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings,
            self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            multimask_output=multimask_output,
        )
        _, num_predictions, low_res_size, _ = low_res_masks.shape

        if output_w > 0 and output_h > 0:
            output_masks = F.interpolate(
                low_res_masks, (output_h, output_w), mode="bicubic"
            )
            output_masks = torch.reshape(
                output_masks,
                (batch_size, max_num_queries, num_predictions, output_h, output_w),
            )
        else:
            output_masks = torch.reshape(
                low_res_masks,
                (
                    batch_size,
                    max_num_queries,
                    num_predictions,
                    low_res_size,
                    low_res_size,
                ),
            )
        iou_predictions = torch.reshape(
            iou_predictions, (batch_size, max_num_queries, num_predictions)
        )
        return output_masks, iou_predictions, low_res_masks


model = build_efficient_sam_vitt()
onnx_model = OnnxEfficientSam(model=model)

output = "weights/efficient_sam_vitt.onnx"

dynamic_axes = {
    "batched_images": {0: "batch", 2: "height", 3: "width"},
    "batched_point_coords": {2: "num_points"},
    "batched_point_labels": {2: "num_points"},
}
dummy_inputs = {
    "batched_images": torch.randn(1, 3, 1024, 1024, dtype=torch.float),
    "batched_point_coords": torch.randint(
        low=0, high=1024, size=(1, 1, 5, 2), dtype=torch.float
    ),
    "batched_point_labels": torch.randint(
        low=0, high=4, size=(1, 1, 5), dtype=torch.float
    ),
}

output_names = ["output_masks", "iou_predictions"]

with open(output, "wb") as f:
    print(f"Exporting onnx model to {output}...")
    torch.onnx.export(
        onnx_model,
        tuple(dummy_inputs.values()),
        f,
        export_params=True,
        verbose=False,
        opset_version=17,
        do_constant_folding=True,
        input_names=list(dummy_inputs.keys()),
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

inference_session = onnxruntime.InferenceSession(output)
output = inference_session.run(
    output_names=None,
    input_feed={k: v.numpy() for k, v in dummy_inputs.items()},
)
