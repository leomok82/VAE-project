import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """
    Convolutional LSTM cell implementation.

    Args:
        input_dim (int): Number of input channels.
        hidden_dim (int): Number of hidden channels.
        kernel_size (tuple): Size of the convolutional kernel.
        bias (bool): Whether to include a bias term in the convolutional layer.

    Attributes:
        input_dim (int): Number of input channels.
        hidden_dim (int): Number of hidden channels.
        kernel_size (tuple): Size of the convolutional kernel.
        padding (tuple): Padding size for the convolutional layer.
        bias (bool): Whether to include a bias term in the convolutional layer.
        conv (nn.Conv2d): Convolutional layer.

    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        """
        Forward pass of the ConvLSTM cell.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, input_dim, height, width).
            cur_state (tuple): Tuple containing the current hidden state and cell state.

        Returns:
            tuple: Tuple containing the next hidden state and cell state.

        """
        h_cur, c_cur = cur_state

        # concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(
            combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        Initialize the hidden state and cell state.

        Args:
            batch_size (int): Batch size.
            image_size (tuple): Size of the input image (height, width).

        Returns:
            tuple: Tuple containing the initial hidden state and cell state.

        """
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    Convolutional LSTM implementation.

    Args:
        input_dim (int): Number of input channels.
        hidden_dim (list): List of hidden channel sizes for each layer.
        kernel_size (list): List of kernel sizes for each layer.
        num_layers (int): Number of layers.
        batch_first (bool): Whether the input tensor has batch size as the first dimension.
        bias (bool): Whether to include a bias term in the convolutional layers.
        return_all_layers (bool): Whether to return outputs and states for all layers.

    Attributes:
        input_dim (int): Number of input channels.
        hidden_dim (list): List of hidden channel sizes for each layer.
        kernel_size (list): List of kernel sizes for each layer.
        num_layers (int): Number of layers.
        batch_first (bool): Whether the input tensor has batch size as the first dimension.
        bias (bool): Whether to include a bias term in the convolutional layers.
        return_all_layers (bool): Whether to return outputs and states for all layers.
        cell_list (nn.ModuleList): List of ConvLSTMCell instances.

    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers, batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError("Inconsistent list length.")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Forward pass of the ConvLSTM.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim, height, width).
            hidden_state (list, optional): List of tuples containing the initial hidden state and cell state for each layer.

        Returns:
            tuple: Tuple containing the output tensor and the last hidden state and cell state for each layer.

        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        """
        Initialize the hidden state and cell state for each layer.

        Args:
            batch_size (int): Batch size.
            image_size (tuple): Size of the input image (height, width).

        Returns:
            list: List of tuples containing the initial hidden state and cell state for each layer.

        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(
                self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """
        Check if the kernel size is consistent.

        Args:
            kernel_size (tuple or list): Kernel size(s) to check.

        Raises:
            ValueError: If the kernel size is not a tuple or a list of tuples.

        """
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        Extend a parameter for multiple layers.

        Args:
            param (int or list): Parameter to extend.
            num_layers (int): Number of layers.

        Returns:
            list: Extended parameter.

        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ConvLSTMModel(nn.Module):
    """
    Convolutional LSTM model implementation.

    Args:
        input_dim (int): Number of input channels.
        hidden_dim (list): List of hidden channel sizes for each layer.
        kernel_size (tuple): Size of the convolutional kernel.
        num_layers (int): Number of layers.
        output_dim (int): Number of output channels.

    Attributes:
        convlstm (ConvLSTM): ConvLSTM instance.
        conv (nn.Conv2d): Convolutional layer.

    """

    def __init__(self, input_dim=1, hidden_dim=[32, 32], kernel_size=(3, 3), num_layers=2, output_dim=1):
        super(ConvLSTMModel, self).__init__()
        self.convlstm = ConvLSTM(input_dim, hidden_dim, kernel_size,
                                 num_layers, batch_first=True, return_all_layers=False)
        self.conv = nn.Conv2d(
            hidden_dim[-1], output_dim, kernel_size=1, padding=0)

    def forward(self, x):
        """
        Forward pass of the ConvLSTM model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim, height, width).

        """
        lstm_out, _ = self.convlstm(x)
        output = self.conv(lstm_out[0][:, -1, :, :, :])
        return output
