/**
 * The smallest max texture size (texels per side) the engine needs: weight
 * textures are as wide as a row's 4-bit values (LFM3's down projection:
 * 7,680; its RGBA4 embedding: 8,192). Not the context: the cache wraps
 * past a texture's height.
 */
export const MIN_TEXTURE_SIZE = 8192
