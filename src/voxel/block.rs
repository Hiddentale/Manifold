#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum BlockType {
    Air,
    Grass,
    Dirt,
    Stone,
    Water,
    Sand,
    Snow,
    Gravel,
}

impl BlockType {
    pub fn is_opaque(self) -> bool {
        !matches!(self, BlockType::Air)
    }
}
