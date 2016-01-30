function pixelCandidates = ColorEnhancement(im)

[In1] = normalize_segmentation(im,'red');
[In2] = normalize_segmentation(im,'blue');
pixelCandidates = logical(In1 + In2);
end