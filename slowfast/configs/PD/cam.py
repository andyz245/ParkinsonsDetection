cam_extractor = GradCAM(model=model, target_layers=[model.s5.pathway0_res2.branch2.c])
out = model(inputs.unsqueeze(0))
activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
result = overlay_mask(to_pil_image(inputs), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
# Display it
plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()
